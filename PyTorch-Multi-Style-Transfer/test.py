import utils
from option import Options
from net import Net, Vgg16
from torch.autograd import Variable
import os
import torch

def main():
    args = Options().parse()
    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    style_loaders = utils.StyleLoader(args.style_folder, args.style_size)
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.style_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()

    content_image = Variable(utils.preprocess_batch(content_image))

    # for i, style_loader in enumerate(style_loaders):
    for i in range(style_loaders.size()):
        print(i)
        style_v = style_loaders.get(i)
        style_model.setTarget(style_v)
        output = style_model(content_image)
        filepath = "out/output"+str(i+1)+'.jpg'
        print(filepath)
        utils.tensor_save_bgrimage(output.data[0], filepath, args.cuda)
    


if __name__ == "__main__":
   main()
