import argparse
import random
from utils import *
from quant import *
import pickle
import time


def get_args_parser():
    parser = argparse.ArgumentParser(description="ERQ-ViT", add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_small', 'vit_base',
                                 'deit_tiny', 'deit_small', 'deit_base',
                                 'swin_tiny', 'swin_small', 'swin_base'],
                        help="model")
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=1024,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')
    parser.add_argument('--coe', default=20000,
                        type=int, help='')

    return parser


model_zoo = {
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224',

    'deit_tiny': 'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base': 'deit_base_patch16_224',

    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224'
}


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def myhook(module, input, output):
    if module.store_input == None:
        module.store_input = []
        module.store_output = []
    module.store_input.append(input[0].cpu().detach())
    module.store_output.append(output.cpu().detach())



def hook_fp_act(q_model, calib_data, args):
    '''
    Args:
        q_model: the quantization model
        cali_data: calibration data
    '''
    # register hook
    hooks = []
    for n, m in q_model.named_modules():
        if isinstance(m, QuantLinear):
            hooks.append(m.register_forward_hook(myhook))
        if isinstance(m, QuantConv2d):
            hooks.append(m.register_forward_hook(myhook))
    # input
    with torch.no_grad():
        _ = q_model(calib_data)

    # remove hook
    for h in hooks:
        h.remove()

    folder_path = f"fp_output/{args.model}-calib{args.calib_batchsize}-W{args.w_bits}A{args.a_bits}"
    os.makedirs(folder_path, exist_ok=True)
    for n, m in q_model.named_modules():
        if isinstance(m, QuantLinear):
            with open(os.path.join(folder_path, n + 'store_input'), 'wb') as file:
                pickle.dump(m.store_input, file)
            m.store_input.clear()
            with open(os.path.join(folder_path, n + 'store_output'), 'wb') as file:
                pickle.dump(m.store_output, file)
            m.store_output.clear()
        if isinstance(m, QuantConv2d):
            with open(os.path.join(folder_path, n + 'store_input'), 'wb') as file:
                pickle.dump(m.store_input, file)
            m.store_input.clear()
            with open(os.path.join(folder_path, n + 'store_output'), 'wb') as file:
                pickle.dump(m.store_output, file)
            m.store_output.clear()

    print("complete collecting fp act...")
    return folder_path

def im2col(input_data, kernel_size, stride, padding):
    # 添加padding
    input_padded = F.pad(input_data, (padding, padding, padding, padding))
    # 获取输入数据的维度
    batch_size, channels, height, width = input_padded.shape

    # 输出的高度和宽度
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1

    # 展开操作
    cols = torch.zeros(batch_size, channels, kernel_size, kernel_size, out_height, out_width, device=input_data.device)

    for y in range(kernel_size):
        y_max = y + stride*out_height
        for x in range(kernel_size):
            x_max = x + stride*out_width
            cols[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.permute(0, 4, 5, 1, 2, 3).reshape(batch_size*out_height*out_width, -1)
    return cols

def conv2d_im2col(input, weight, bias=None, stride=1, padding=0):
    # 卷积核参数
    kernel_size = weight.shape[2]
    # 计算输出的高度和宽度
    out_height = (input.shape[2] + 2 * padding - kernel_size) // stride + 1
    out_width = (input.shape[3] + 2 * padding - kernel_size) // stride + 1
    # 展开输入
    cols = im2col(input, kernel_size, stride, padding)
    # 权重矩阵化
    weights_col = weight.reshape(weight.shape[0], -1).T
    # 矩阵乘法
    print(cols.shape, weights_col.shape)
    output = torch.matmul(cols, weights_col)
    if bias is not None:
        output += bias.view(-1)
    output = output.reshape(input.shape[0], out_height, out_width, weight.shape[0]).permute(0, 3, 1, 2)
    return output

def main():
    print(args)
    seed(args.seed)

    device = torch.device(args.device)

    # Build dataloader
    print('Building dataloader ...')
    train_loader, val_loader = build_dataset(args)
    for data, target in train_loader:
        calib_data = data.to(device)
        target = target.to(device)
        break

    # Build model
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()



    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    @torch.no_grad()
    def reparameterization(q_model):
        # Scale reparameterization
        print('Performing scale reparameterization ...')
        with torch.no_grad():
            module_dict = {}
            q_model_slice = q_model.layers if 'swin' in args.model else q_model.blocks
            for name, module in q_model_slice.named_modules():
                module_dict[name] = module
                idx = name.rfind('.')
                if idx == -1:
                    idx = 0
                father_name = name[:idx]
                if father_name in module_dict:
                    father_module = module_dict[father_name]
                else:
                    raise RuntimeError(f"father module {father_name} not found")
                # if 'norm1' in name or 'norm2' in name or 'norm' in name:
                if 'norm1' in name or 'norm2' in name:
                    if 'norm1' in name:
                        next_module = father_module.attn.qkv
                    elif 'norm2' in name:
                        next_module = father_module.mlp.fc1
                    else:
                        next_module = father_module.reduction

                    act_delta = next_module.input_quantizer.delta.reshape(-1)
                    act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                    act_min = -act_zero_point * act_delta

                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta

                    r = act_delta / target_delta
                    b = act_min / r - target_min

                    module.weight.data = module.weight.data / r
                    module.bias.data = module.bias.data / r - b

                    next_module.weight.data = next_module.weight.data * r
                    if next_module.bias is not None:
                        next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data,
                                                                                 b.reshape(-1, 1)).reshape(-1)
                    else:
                        next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                        next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1)

                    next_module.input_quantizer.channel_wise = False
                    next_module.input_quantizer.delta = Parameter(target_delta).contiguous()
                    next_module.input_quantizer.zero_point = Parameter(target_zero_point).contiguous()
                    next_module.weight_quantizer.inited.fill_(0)

    @torch.no_grad()
    def replace_W(q_model, folder_path):

        for n, m in q_model.named_modules():
            if isinstance(m, QuantLinear):
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")
                fp_input = store_input[0]
                if len(fp_input.shape) == 2:
                    num_of_inverse = 0.1
                    print('num_of_inverse', num_of_inverse)
                else:
                    num_of_inverse = 1e-1 * args.coe
                    print('num_of_inverse', num_of_inverse)


                fp_output_shape = store_output[0].shape
                fp_output_flat = store_output[0].cuda().reshape(-1, fp_output_shape[-1])
                quan_output = m.input_quantizer(store_input[0].cuda())
                del store_input

                w = m.weight.clone()

                if getattr(m, "bias") is not None:
                    print('bias!')
                    b = m.bias.clone()
                    W_cat = torch.cat((w, b.unsqueeze(1)), dim=1).cuda()

                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                    quan_output_cat = torch.cat((quan_output_flat, torch.ones(quan_output_flat.shape[0], 1).cuda()),
                                                dim=1)

                    A = quan_output_cat
                    Y = fp_output_flat - (quan_output_cat @ W_cat.T)

                    beta = torch.inverse(A.permute(1, 0) @ A
                                         + torch.eye(A.shape[1]).cuda() * num_of_inverse) @ A.permute(1, 0) @ Y

                    new_W, new_b_0 = torch.split(beta, [beta.shape[0] - 1, 1], dim=0)  # split on the output channel
                    new_b = new_b_0.squeeze()
                    m.weight.data = new_W.T + w
                    m.bias.data = new_b + b

                    del fp_output_flat, quan_output, w, b, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                        beta, new_W, new_b_0, new_b
                    torch.cuda.empty_cache()  # 清除未使用的缓存
                else:
                    print('None bias!')
                    W_cat = w.cuda()
                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                    quan_output_cat = quan_output_flat

                    A = quan_output_cat
                    Y = fp_output_flat - (quan_output_cat @ W_cat.T)

                    beta = torch.inverse(A.permute(1, 0) @ A
                                         + torch.eye(A.shape[1]).cuda() * num_of_inverse) @ A.permute(1, 0) @ Y

                    new_W = beta
                    m.weight.data = new_W.T + w

                    del fp_output_flat, quan_output, w, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                        beta, new_W
                    torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()


            if isinstance(m, QuantConv2d):
                if 'embed' in n:
                    print('skip QuantConv2d!')
                    continue
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")
                quan_output = m.input_quantizer(store_input[0].cuda())

                # 卷积核参数
                kernel_size = m.weight.shape[2]
                stride = m.stride[0]
                padding = m.padding[0]
                quan_output_cols = im2col(quan_output, kernel_size, stride, padding)

                # 权重矩阵化
                weights_col = deepcopy(m.weight.reshape(m.weight.shape[0], -1).T)

                del store_input

                num_of_inverse = 1e-1 * args.coe
                print('num_of_inverse', num_of_inverse)

                with torch.no_grad():
                    w = weights_col

                    if getattr(m, "bias") is not None:
                        print('bias!')

                        b = m.bias.clone()
                        W_cat = torch.cat((w, b.unsqueeze(0)), dim=0).cuda()

                        quan_output_flat = quan_output_cols
                        quan_output_cat = torch.cat((quan_output_flat,
                                                     torch.ones(quan_output_flat.shape[0], 1).cuda()),
                                                    dim=1)

                        A = quan_output_cat
                        tmp = (quan_output_cat @ W_cat)
                        fp_output_flat = store_output[0].cuda()
                        fp_output_flat = fp_output_flat.permute(0, 2, 3, 1).reshape(tmp.shape)
                        Y = fp_output_flat - tmp
                        beta = torch.inverse(A.permute(1, 0) @ A
                                             + torch.eye(A.shape[1]).cuda() * num_of_inverse) @ A.permute(1, 0) @ Y

                        new_W, new_b_0 = torch.split(beta, [beta.shape[0] - 1, 1], dim=0)  # split on the output channel
                        new_b = new_b_0.squeeze()

                        m.weight.data = (new_W + w).T.reshape(m.weight.shape)
                        m.bias.data = new_b + b

                        # judge whether the mse decend

                        del fp_output_flat, quan_output, w, b, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                            beta, new_W, new_b_0, new_b
                        torch.cuda.empty_cache()  # 清除未使用的缓存
                    else:
                        print('None bias!')
                        W_cat = w.cuda()

                        quan_output_flat = quan_output_cols

                        A = quan_output_cat
                        tmp = (quan_output_cat @ W_cat)
                        fp_output_flat = store_output[0].cuda()
                        fp_output_flat = fp_output_flat.permute(0, 2, 3, 1).reshape(tmp.shape)
                        Y = fp_output_flat - tmp
                        beta = torch.inverse(A.permute(1, 0) @ A
                                             + torch.eye(A.shape[1]).cuda() * num_of_inverse) @ A.permute(1, 0) @ Y

                        new_W = beta

                        m.weight.data = (new_W + w).T.reshape(m.weight.shape)


                        del fp_output_flat, quan_output, w, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                            beta, new_W
                        torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()

        return

    @torch.no_grad()
    def replace_W_afterquant_vector_twopart(q_model, folder_path, args):

        for n, m in q_model.named_modules():
            if isinstance(m, QuantLinear):
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")
                fp_input = store_input[0]

                if len(fp_input.shape) == 2:
                    num_of_inverse = 0.1
                    print('num_of_inverse', num_of_inverse)
                else:
                    num_of_inverse = 1e-1 * args.coe
                    print('num_of_inverse', num_of_inverse)

                quan_output = m.input_quantizer(store_input[0].cuda())
                # del store_input

                num_of_inverse = 1e-1 * args.coe
                print('num_of_inverse', num_of_inverse)

                if getattr(m, "bias") is not None:
                    print('bias!')
                    w = m.weight.clone()
                    b = m.bias.clone()

                    print(f'redistribute W, there are {w.shape[0]} output channel of layer {n}')

                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                    quan_output_flat = torch.cat(
                        (quan_output_flat, torch.ones((quan_output_flat.shape[0], 1)).cuda()), dim=1)

                    current = torch.cat((w, b.clone().unsqueeze(1)), dim=1).detach().clone().cuda()
                    mask = torch.ones_like(current[0]).bool()
                    while torch.sum(mask) > 1:

                        number_of_quant = torch.sum(mask) // 2
                        number_of_adjust = torch.sum(mask) - number_of_quant

                        x_dequant_floor = m.weight_quantizer(current, 'floor')
                        w_error_floor = x_dequant_floor - current
                        w_error_floor[:, -1] = 0

                        x_dequant_ceil = m.weight_quantizer(current, 'ceil')
                        w_error_ceil = x_dequant_ceil - current
                        w_error_ceil[:, -1] = 0

                        x_dequant = m.weight_quantizer(current, 'round')
                        w_error = x_dequant - current
                        w_error[:, -1] = 0

                        outlier_indices = torch.arange(0+torch.sum(~mask), number_of_quant+torch.sum(~mask))

                        if args.model == 'swin_small' or args.model == 'swin_tiny':
                            B = 500
                        elif args.model == 'swin_base':
                            B = 100
                        else:
                            B = 500

                        ###
                        # if ('.mlp.fc1' in n or '.attn.qkv' in n):
                        if True:
                            means = torch.mean(quan_output_flat[:, outlier_indices], dim=0)
                            covs = torch.cov(quan_output_flat[:, outlier_indices].T)
                            coes = means.unsqueeze(0).T @ means.unsqueeze(0) + covs

                            groups = math.ceil(len(current) / B)
                            for g in range(groups):

                                a = np.arange(g * B, min((g + 1) * B, len(current)))
                                current_outputs = torch.tensor(a).cuda()


                                sub_delta1 = w_error.clone()[current_outputs][:, outlier_indices]
                                sub_delta_floor = w_error_floor[current_outputs][:, outlier_indices]
                                sub_delta_ceil = w_error_ceil[current_outputs][:, outlier_indices]

                                fail_dim = torch.zeros(len(sub_delta1)).bool()
                                count = 0
                                while count < 100 and torch.sum(~fail_dim) > 0:
                                    count += 1

                                    gradient = (2 * coes @ sub_delta1.T).T
                                    same_sign = (sub_delta1 * gradient > 0)
                                    gradient[~same_sign] = 0  # find these para that need to be change sign
                                    gradient[:, ~mask[outlier_indices]] = 0

                                    number_of_nonzero_gradi = torch.sum(gradient != 0, dim=1)
                                    number_of_flip = torch.minimum(number_of_nonzero_gradi, torch.tensor(1))
                                    _, max_diff_indexs = torch.topk(abs(gradient), k=int(torch.max(number_of_flip).item()), dim=1)

                                    v = torch.gather(sub_delta1, 1, max_diff_indexs) - torch.gather(gradient, 1, max_diff_indexs)
                                    ceils = torch.gather(sub_delta_ceil, 1, max_diff_indexs)
                                    floors = torch.gather(sub_delta_floor, 1, max_diff_indexs)
                                    distance_to_ceil = torch.abs(v - ceils)
                                    distance_to_floor = torch.abs(v - floors)
                                    v = torch.where(distance_to_ceil <= distance_to_floor, ceils, floors)

                                    cur_min = (sub_delta1.unsqueeze(1) @ coes @ sub_delta1.unsqueeze(2)).squeeze()
                                    tmp = torch.gather(sub_delta1, 1, max_diff_indexs).clone()
                                    sub_delta1.scatter_(1, max_diff_indexs, v)

                                    cur_min_v = (sub_delta1.unsqueeze(1) @ coes @ sub_delta1.unsqueeze(2)).squeeze()

                                    fail_dim = cur_min_v > cur_min

                                    temp = sub_delta1[fail_dim].clone()
                                    temp.scatter_(1, max_diff_indexs[fail_dim], tmp[fail_dim])
                                    sub_delta1[fail_dim] = temp

                                w_error[current_outputs.unsqueeze(1), outlier_indices] = sub_delta1
                        ###

                        mask[outlier_indices] = False
                        remaining_indices = torch.nonzero(mask).squeeze()
                        non_outliers_indices = remaining_indices
                        groups = math.ceil(len(current) / B)

                        for g in range(groups):
                            current_outputs = torch.arange(g * B, min((g + 1) * B, len(current))).cuda()

                            w1 = current[current_outputs][:, outlier_indices]
                            w2 = current[current_outputs][:, non_outliers_indices]

                            I1 = quan_output_flat[:, outlier_indices]
                            I2 = quan_output_flat[:, non_outliers_indices]
                            delta1 = w_error[current_outputs][:, outlier_indices]
                            delta2 = -torch.inverse(
                                I2.T @ I2 + num_of_inverse * torch.eye(number_of_adjust).cuda()) @ (
                                             I2.T @ I1) @ delta1.T
                            w2 += delta2.T

                            if len(w2.shape) == 1:
                                w2 = w2.unsqueeze(1)

                            current[current_outputs.unsqueeze(1), outlier_indices] = w1 + delta1
                            current[current_outputs.unsqueeze(1), non_outliers_indices] = w2


                    new_w, new_b = torch.split(current, [current.shape[1] - 1, 1], dim=1)

                    w.copy_(new_w)
                    b.copy_(new_b.squeeze())

                    m.weight.data = w
                    m.bias.data = b
                    m.set_quant_state(True, False)
                    torch.cuda.empty_cache()

                else:
                    w = m.weight.clone()

                    print(f'redistribute W, there are {w.shape[0]} output channel of layer {n}')

                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])

                    current = w.detach().clone().cuda()
                    mask = torch.ones_like(current[0]).bool()
                    while torch.sum(mask) > 1:

                        number_of_quant = torch.sum(mask) // 2
                        number_of_adjust = torch.sum(mask) - number_of_quant

                        x_dequant_floor = m.weight_quantizer(current, 'floor')
                        w_error_floor = x_dequant_floor - current
                        w_error_floor[:, -1] = 0

                        x_dequant_ceil = m.weight_quantizer(current, 'ceil')
                        w_error_ceil = x_dequant_ceil - current
                        w_error_ceil[:, -1] = 0

                        x_dequant = m.weight_quantizer(current, 'round')
                        w_error = x_dequant - current
                        w_error[:, -1] = 0

                        outlier_indices = torch.arange(0+torch.sum(~mask), number_of_quant+torch.sum(~mask))

                        if args.model == 'swin_small' or args.model == 'swin_tiny':
                            B = 500
                        elif args.model == 'swin_base':
                            B = 100
                        else:
                            B = 500

                        ###
                        # if ('.mlp.fc1' in n or '.attn.qkv' in n):
                        if True:
                            means = torch.mean(quan_output_flat[:, outlier_indices], dim=0)
                            covs = torch.cov(quan_output_flat[:, outlier_indices].T)
                            coes = means.unsqueeze(0).T @ means.unsqueeze(0) + covs

                            groups = math.ceil(len(current) / B)
                            for g in range(groups):

                                a = np.arange(g * B, min((g + 1) * B, len(current)))
                                current_outputs = torch.tensor(a).cuda()


                                sub_delta1 = w_error.clone()[current_outputs][:, outlier_indices]
                                sub_delta_floor = w_error_floor[current_outputs][:, outlier_indices]
                                sub_delta_ceil = w_error_ceil[current_outputs][:, outlier_indices]

                                fail_dim = torch.zeros(len(sub_delta1)).bool()
                                count = 0
                                while count < 100 and torch.sum(~fail_dim) > 0:
                                    count += 1

                                    gradient = (2 * coes @ sub_delta1.T).T
                                    same_sign = (sub_delta1 * gradient > 0)
                                    gradient[~same_sign] = 0  # find these para that need to be change sign
                                    gradient[:, ~mask[outlier_indices]] = 0

                                    number_of_nonzero_gradi = torch.sum(gradient != 0, dim=1)
                                    number_of_flip = torch.minimum(number_of_nonzero_gradi, torch.tensor(1))
                                    _, max_diff_indexs = torch.topk(abs(gradient), k=int(torch.max(number_of_flip).item()), dim=1)

                                    v = torch.gather(sub_delta1, 1, max_diff_indexs) - torch.gather(gradient, 1, max_diff_indexs)
                                    ceils = torch.gather(sub_delta_ceil, 1, max_diff_indexs)
                                    floors = torch.gather(sub_delta_floor, 1, max_diff_indexs)
                                    distance_to_ceil = torch.abs(v - ceils)
                                    distance_to_floor = torch.abs(v - floors)
                                    v = torch.where(distance_to_ceil <= distance_to_floor, ceils, floors)

                                    cur_min = (sub_delta1.unsqueeze(1) @ coes @ sub_delta1.unsqueeze(2)).squeeze()
                                    tmp = torch.gather(sub_delta1, 1, max_diff_indexs).clone()
                                    sub_delta1.scatter_(1, max_diff_indexs, v)

                                    cur_min_v = (sub_delta1.unsqueeze(1) @ coes @ sub_delta1.unsqueeze(2)).squeeze()

                                    fail_dim = cur_min_v > cur_min

                                    temp = sub_delta1[fail_dim].clone()
                                    temp.scatter_(1, max_diff_indexs[fail_dim], tmp[fail_dim])
                                    sub_delta1[fail_dim] = temp

                                w_error[current_outputs.unsqueeze(1), outlier_indices] = sub_delta1
                        ###

                        mask[outlier_indices] = False
                        remaining_indices = torch.nonzero(mask).squeeze()
                        non_outliers_indices = remaining_indices
                        groups = math.ceil(len(current) / B)

                        for g in range(groups):
                            current_outputs = torch.arange(g * B, min((g + 1) * B, len(current))).cuda()

                            w1 = current[current_outputs][:, outlier_indices]
                            w2 = current[current_outputs][:, non_outliers_indices]

                            I1 = quan_output_flat[:, outlier_indices]
                            I2 = quan_output_flat[:, non_outliers_indices]
                            delta1 = w_error[current_outputs][:, outlier_indices]
                            delta2 = -torch.inverse(
                                I2.T @ I2 + num_of_inverse * torch.eye(number_of_adjust).cuda()) @ (
                                             I2.T @ I1) @ delta1.T
                            w2 += delta2.T

                            if len(w2.shape) == 1:
                                w2 = w2.unsqueeze(1)

                            current[current_outputs.unsqueeze(1), outlier_indices] = w1 + delta1
                            current[current_outputs.unsqueeze(1), non_outliers_indices] = w2


                    new_w = current

                    w.copy_(new_w)
                    import IPython
                    IPython.embed()

                    m.weight.data = w
                    m.set_quant_state(True, False)
                    torch.cuda.empty_cache()

                    '''
                    print('None bias!')
                    for i, _ in enumerate(range(w.shape[0])):
                        print(f'redistribute W of {i}/{w.shape[0]} output channel of layer {n}')

                        quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                        current = w[i, :].clone().detach().cuda()

                        mask = torch.ones_like(current).bool()

                        while torch.sum(mask) > 1:
                            number_of_quant = torch.sum(mask) // 2
                            number_of_adjust = torch.sum(mask) - number_of_quant

                            x_dequant_floor = m.weight_quantizer(current, 'floor', i)
                            w_error_floor = x_dequant_floor - current

                            x_dequant_ceil = m.weight_quantizer(current, 'ceil', i)
                            w_error_ceil = x_dequant_ceil - current

                            x_dequant = m.weight_quantizer(current, 'round', i)
                            w_error = x_dequant - current

                            w_error[~mask] += torch.inf
                            w_error[-1] += torch.inf
                            _, outlier_indices = torch.topk(-torch.abs(w_error), number_of_quant)

                            if ('.mlp.fc1' in n or '.attn.qkv' in n):
                            # if True:
                                means = torch.mean(quan_output_flat[:, outlier_indices], dim=0)
                                covs = torch.cov(quan_output_flat[:, outlier_indices].T)
                                coes = means.unsqueeze(0).T @ means.unsqueeze(0) + covs

                                sub_delta1 = w_error.clone()[outlier_indices]
                                sub_delta_floor = w_error_floor[:-1][outlier_indices]
                                sub_delta_ceil = w_error_ceil[:-1][outlier_indices]
                                sub_indicators = (sub_delta1 == sub_delta_floor).int()

                                count = 0
                                while count < 100:
                                    count += 1
                                    gradient = 2 * coes @ sub_delta1
                                    same_sign = (sub_delta1 * gradient > 0)
                                    gradient[~same_sign] = 0  # find these para that need to be change sign
                                    gradient[~mask[outlier_indices]] = 0

                                    number_of_nonzero_gradi = torch.sum(gradient != 0)
                                    number_of_flip = min(number_of_nonzero_gradi, 1)
                                    _, max_diff_indexs = torch.topk(abs(gradient), number_of_flip)

                                    v = sub_delta1[max_diff_indexs] - gradient[max_diff_indexs]
                                    distance_to_ceil = torch.abs(v - sub_delta_ceil[max_diff_indexs])
                                    distance_to_floor = torch.abs(v - sub_delta_floor[max_diff_indexs])
                                    v = torch.where(distance_to_ceil <= distance_to_floor,
                                                    sub_delta_ceil[max_diff_indexs], sub_delta_floor[max_diff_indexs])

                                    cur_min = sub_delta1.T @ coes @ sub_delta1
                                    tmp = sub_delta1[max_diff_indexs].clone()
                                    sub_delta1[max_diff_indexs] = v
                                    cur_min_v = sub_delta1.T @ coes @ sub_delta1
                                    if cur_min_v >= cur_min:
                                        sub_delta1[max_diff_indexs] = tmp
                                        break
                                    w_error[:-1] = sub_delta1

                            mask[outlier_indices] = False
                            remaining_indices = torch.nonzero(mask).squeeze()
                            non_outliers_indices = remaining_indices

                            w1 = current[outlier_indices]
                            w2 = current[non_outliers_indices]

                            I1 = quan_output_flat[:, outlier_indices]
                            I2 = quan_output_flat[:, non_outliers_indices]
                            delta1 = w_error[outlier_indices]
                            delta2 = -torch.inverse(
                                I2.T @ I2 + num_of_inverse * torch.eye(number_of_adjust).cuda()) @ (
                                             I2.T @ I1) @ delta1
                            w2 += delta2
                            current[outlier_indices] = w1 + delta1
                            current[non_outliers_indices] = w2


                        x_dequant = m.weight_quantizer(current, 'round', i)
                        remaining_indices = torch.nonzero(mask).squeeze()
                        current[remaining_indices] = x_dequant[remaining_indices]

                        new_w = current
                        print('max w[i, :]', torch.max(w[i, :]), 'min w[i, :]', torch.min(w[i, :]),
                              'max new_w', torch.max(new_w), 'min new_w', torch.min(new_w), )
                        w[i, :].copy_(new_w)
                    '''


                    m.weight.data = w
                    m.set_quant_state(True, False)
                    torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()

            if isinstance(m, QuantConv2d):
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")

                if hasattr(m, 'input_quantizer'):
                    quan_output = m.input_quantizer(store_input[0].cuda())
                else:
                    quan_output = store_input[0].cuda()

                # 卷积核参数
                kernel_size = m.weight.shape[2]
                stride = m.stride[0]
                padding = m.padding[0]
                quan_output_cols = im2col(quan_output, kernel_size, stride, padding)

                # 权重矩阵化
                weights_col = deepcopy(m.weight.reshape(m.weight.shape[0], -1).T)

                del store_input

                num_of_inverse = 1e-1 * args.coe * 10
                print('num_of_inverse', num_of_inverse)

                # if False:
                if getattr(m, "bias") is not None:

                    print('bias!')
                    w = weights_col
                    b = m.bias.clone()
                    print(f'redistribute W, there are {w.shape[0]} output channel of layer {n}')
                    quan_output_flat = quan_output_cols
                    quan_output_flat = torch.cat((quan_output_flat,
                                                 torch.ones(quan_output_flat.shape[0], 1).cuda()),
                                                dim=1)

                    current = torch.cat((w, b.unsqueeze(0)), dim=0).detach().clone().cuda()
                    # current = current.unsqueeze(0).unsqueeze(0)
                    current = current.T
                    mask = torch.ones_like(current[0]).bool()

                    while torch.sum(mask) > 1:

                        number_of_quant = torch.sum(mask) // 2
                        number_of_adjust = torch.sum(mask) - number_of_quant

                        current = current.unsqueeze(2).unsqueeze(2)
                        x_dequant = m.weight_quantizer(current, 'round')
                        w_error = x_dequant - current
                        w_error[:, -1] = 0
                        w_error = w_error.squeeze()
                        current = current.squeeze()

                        outlier_indices = torch.arange(0+torch.sum(~mask), number_of_quant+torch.sum(~mask))

                        if args.model == 'swin_small' or args.model == 'swin_tiny':
                            B = 500
                        elif args.model == 'swin_base':
                            B = 100
                        else:
                            B = 500

                        mask[outlier_indices] = False
                        remaining_indices = torch.nonzero(mask).squeeze()
                        non_outliers_indices = remaining_indices
                        groups = math.ceil(len(current) / B)

                        for g in range(groups):

                            current_outputs = torch.arange(g * B, min((g + 1) * B, len(current))).cuda()

                            w1 = current[current_outputs][:, outlier_indices]
                            w2 = current[current_outputs][:, non_outliers_indices]

                            I1 = quan_output_flat[:, outlier_indices]
                            I2 = quan_output_flat[:, non_outliers_indices]
                            delta1 = w_error[current_outputs][:, outlier_indices]
                            delta2 = -torch.inverse(
                                I2.T @ I2 + num_of_inverse * torch.eye(number_of_adjust).cuda()) @ (
                                             I2.T @ I1) @ delta1.T
                            w2 += delta2.T

                            if len(w2.shape) == 1:
                                w2 = w2.unsqueeze(1)

                            current[current_outputs.unsqueeze(1), outlier_indices] = w1 + delta1
                            current[current_outputs.unsqueeze(1), non_outliers_indices] = w2

                    new_w, new_b = torch.split(current, [current.shape[1] - 1, 1], dim=1)
                    new_b = new_b.squeeze()

                    m.weight.data = new_w.reshape(m.weight.shape)
                    m.bias.data = new_b

                    m.set_quant_state(True, False)
                    torch.cuda.empty_cache()

                else:
                    w = m.weight.clone()
                    print('None bias!')
                    for i, _ in enumerate(range(w.shape[0])):
                        print(f'redistribute W of {i}/{w.shape[0]} output channel of layer {n}')

                        quan_output_flat = quan_output_cols
                        current = w[i, :].clone().detach().cuda()
                        current = current.reshape(-1)
                        mask = torch.ones_like(current).bool()
                        mask = mask.reshape(-1)

                        while torch.sum(mask) > 1:

                            number_of_quant = torch.sum(mask) // 2
                            number_of_adjust = torch.sum(mask) - number_of_quant

                            x_dequant = m.weight_quantizer(current, 'round', i)
                            w_error = x_dequant - current
                            w_error = w_error.reshape(-1)

                            w_error[~mask] += torch.inf
                            w_error[-1] += torch.inf
                            _, outlier_indices = torch.topk(-torch.abs(w_error), number_of_quant)


                            mask[outlier_indices] = False
                            remaining_indices = torch.nonzero(mask).squeeze()
                            non_outliers_indices = remaining_indices

                            w1 = current[outlier_indices]
                            w2 = current[non_outliers_indices]

                            I1 = quan_output_flat[:, outlier_indices]
                            I2 = quan_output_flat[:, non_outliers_indices]
                            delta1 = w_error[outlier_indices]

                            delta2 = -torch.inverse(
                                I2.T @ I2 + num_of_inverse * torch.eye(number_of_adjust).cuda()) @ (
                                             I2.T @ I1) @ delta1
                            w2 += delta2
                            current[outlier_indices] = w1 + delta1
                            current[non_outliers_indices] = w2

                        x_dequant = m.weight_quantizer(current, 'round', i)
                        x_dequant = x_dequant.reshape(-1)
                        remaining_indices = torch.nonzero(mask).squeeze()
                        current[remaining_indices] = x_dequant[remaining_indices]

                        new_w = current
                        print('max w[i, :]', torch.max(w[i, :]), 'min w[i, :]', torch.min(w[i, :]),
                              'max new_w', torch.max(new_w), 'min new_w', torch.min(new_w), )
                        w[i, :].copy_(new_w.reshape(w[i, :].shape))

                    m.weight.data = w
                    m.set_quant_state(True, False)

                    torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()

        return


    start_time = time.time()
    # Initial quantization
    print('Performing initial quantization of Act ...')
    set_quant_state(q_model, input_quant=True, weight_quant=False)
    with torch.no_grad():
        _ = q_model(calib_data[:32])


    # computer W by fp-act and quant-act
    set_quant_state(q_model, input_quant=True, weight_quant=False)
    # repar first
    reparameterization(q_model)
    ## store fp output
    set_quant_state(q_model, input_quant=False, weight_quant=False)
    fp_folder_path = hook_fp_act(q_model, calib_data, args)


    set_quant_state(q_model, input_quant=True, weight_quant=False)
    replace_W(q_model, fp_folder_path)

    # Re-calibration
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)
    with torch.no_grad():
        print('acc of calib_data1:')
        a = torch.sum(torch.argmax(q_model(calib_data), dim=1) == target) / len(target)
        print(a.cpu().numpy())

    replace_W_afterquant_vector_twopart(q_model, fp_folder_path, args)

    #
    set_quant_state(q_model, input_quant=True, weight_quant=True)


    end_time = time.time()
    execution_time = end_time - start_time
    print("runing time of execution_time: ", execution_time, "s")
    with torch.no_grad():
        print('acc of calib_data2:')
        a = torch.sum(torch.argmax(q_model(calib_data), dim=1) == target) / len(target)
        print(a.cpu().numpy())
    # Validate the quantized model
    print("Validating ...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, q_model, criterion, device
    )


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


def test(args, data, target, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()

    data = data.to(device)
    target = target.to(device)

    with torch.no_grad():
        output = model(data)
    loss = criterion(output, target)

    # Measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data.item(), data.size(0))
    top1.update(prec1.data.item(), data.size(0))
    top5.update(prec5.data.item(), data.size(0))

    # Measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum()
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ERQ', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
