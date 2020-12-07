## 环境准备
### 替换vgg模型
由于vgg作者原来加载的tourch7的vgg16.t7不匹配了。

替换成pytorch的vgg模型，替换的原理是vgg是用来抽取图像特征的。所以添加自己的

修改了以下代码。
```shell
--- a/experiments/main.py
+++ b/experiments/main.py
@@ -26,6 +26,7 @@ import utils
 from net import Net, Vgg16
 
 from option import Options
+import torch.nn as nn
 
 def main():
     # figure out the experiments type
@@ -51,6 +52,23 @@ def main():
     else:
         raise ValueError('Unknow experiment type')
 
+from torchvision import models
+
+class VGGNet(nn.Module):
+    def __init__(self):
+        """Select conv1_1 ~ conv5_1 activation maps."""
+        super(VGGNet, self).__init__()
+        self.select = ['0', '5', '10', '19'] 
+        self.vgg = models.vgg19(pretrained=True).features
+        
+    def forward(self, x):
+        """Extract multiple convolutional feature maps."""
+        features = []
+        for name, layer in self.vgg._modules.items():
+            x = layer(x)
+            if name in self.select:
+                features.append(x)
+        return features
 
 def optimize(args):
     """    Gatys et al. CVPR 2017
@@ -67,9 +85,12 @@ def optimize(args):
     style_image = utils.subtract_imagenet_mean_batch(style_image)
 
     # load the pre-trained vgg-16 and extract features
-    vgg = Vgg16()
-    utils.init_vgg16(args.vgg_model_dir)
-    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
+    # vgg = Vgg16()
+    # utils.init_vgg16(args.vgg_model_dir)
+    # vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
+    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
+    vgg = VGGNet().to(device).eval()
+
     if args.cuda:
         content_image = content_image.cuda()
         style_image = style_image.cuda()
@@ -107,6 +128,7 @@ def optimize(args):
 
 def train(args):
     check_paths(args)
+    print(args)
     np.random.seed(args.seed)
     torch.manual_seed(args.seed)
 
@@ -131,10 +153,11 @@ def train(args):
     optimizer = Adam(style_model.parameters(), args.lr)
     mse_loss = torch.nn.MSELoss()
 
-    vgg = Vgg16()
-    utils.init_vgg16(args.vgg_model_dir)
-    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
-
+    # vgg = Vgg16()
+    # utils.init_vgg16(args.vgg_model_dir)
+    # vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
+    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
+    vgg = VGGNet().cuda().eval()
     if args.cuda:
         style_model.cuda()
         vgg.cuda()
@@ -185,8 +208,8 @@ def train(args):
             total_loss.backward()
             optimizer.step()
 
-            agg_content_loss += content_loss.data[0]
-            agg_style_loss += style_loss.data[0]
+            agg_content_loss += content_loss.item()
+            agg_style_loss += style_loss.item()
```

然后模型会自动下载到'~/.cache/torch/hub/checkpoints',你也可以先提前下载好，直接放进去。
### 训练数据准备
将voc的数据放到dataset目录
```shell
        experiments/dataset/train2014.zip
        experiments/dataset/train2014/
        experiments/dataset/val2014.zip
        experiments/dataset/val2014/

```

## 启动训练

```shell
python main.py train --epochs 4

```

## 启动测试
```shell
python test.py test --model ./models/Epoch_0iters_120000_Mon_Nov_30_16:58:56_2020_1.0_5.0.model
```

## 展示

<img src ="pic/output1.jpg" width="260px" /> 
<img src ="pic/output2.jpg" width="260px" />
<img src ="pic/output3.jpg" width="260px" />
<img src ="pic/output4.jpg" width="260px" />
<img src ="pic/output5.jpg" width="260px" />
<img src ="pic/output6.jpg" width="260px" />
<img src ="pic/output7.jpg" width="260px" />
<img src ="pic/output8.jpg" width="260px" />
<img src ="pic/output9.jpg" width="260px" />