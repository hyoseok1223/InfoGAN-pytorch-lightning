import yaml
import torch
import numpy as np

def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        x = x.unsqueeze(1) # 이게 필요함.
        c.scatter_(1, x, 1) # dim, index, src value
    return c

def sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=None, supervised=False):
    z = torch.randn(batch_size, n_noise)#.to(DEVICE)
    
    if supervised:
        c_discrete = to_onehot(label)#.to(DEVICE) # (B,10)
    else:
        c_discrete = to_onehot(torch.LongTensor(batch_size, 1).random_(0, n_c_discrete))#.to(DEVICE) # (B,10)
   
    # uniform ( Rotation & Width)
    c_continuous = torch.zeros(batch_size, n_c_continuous).uniform_(-1, 1)#.to(DEVICE) # (B,2)
    
    c = torch.cat((c_discrete.float(), c_continuous), 1)
    return z, c

def get_sample_image(model,args):
    """
        save sample 100 images
    """
    images = []
    # 각 code정보에 따라서 이미지들이 생성될 수 있도록 짜여진 함수

    # continuous code
    for cc_type in range(2): # 2 continous code
        for num in range(10):
            fix_z = torch.randn(1, args.n_noise)
            z = fix_z.to(args.DEVICE)
            cc = -1
            for i in range(10):
                cc += 0.2
                c_discrete = to_onehot(num).to(args.DEVICE) # (B,10)
                c_continuous = torch.zeros(1, args.n_c_continuous).to(args.DEVICE)
                c_continuous.data[:,cc_type].add_(cc)
                c = torch.cat((c_discrete.float(), c_continuous), 1)
                y_hat = model(z, c)
                line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
            all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
        img = all_img.cpu().data.numpy()
        images.append(img)

    # discrete code
    for num in range(10):
        c_discrete = to_onehot(num).to(args.DEVICE) # (B,10)
        for i in range(10):
            z = torch.randn(1, args.n_noise).to(args.DEVICE)
            c_continuous = torch.zeros(1, args.n_c_continuous).to(args.DEVICE)
            c = torch.cat((c_discrete.float(), c_continuous), 1)
            y_hat = model(z, c)
            line_img = torch.cat((line_img, y_hat.view(28, 28)), dim=1) if i > 0 else y_hat.view(28, 28)
        all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
    img = all_img.cpu().data.numpy()
    images.append(img)
    return images[0], images[1], images[2]

def log_gaussian(c, mu, var):
    """
        criterion for Q(condition classifier)
        reparameterization trick
    """
    return -((c - mu)**2)/(2*var+1e-8) - 0.5*torch.log(2*np.pi*var+1e-8)

def load_config(src_config_dir):
    with open(src_config_dir ,'r') as f:
        config = yaml.safe_load(f)
    return config
