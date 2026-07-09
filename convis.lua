-- Modified from github.com/htoyryla's code here: https://github.com/htoyryla/convis/issues/2#issuecomment-365009705

require 'torch'
require 'nn'
require 'image'
require 'loadcaffe' 

function preprocess(img)
   local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
   local perm = torch.LongTensor{3, 2, 1}
   img = img:index(1, perm):mul(256.0)
   mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
   img:add(-1, mean_pixel)
   return img
end

function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

local cmd = torch.CmdLine()

cmd:option('-content_image', 'examples/inputs/tubingen.jpg')
cmd:option('-image_size', 800, 'output image size')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-layer', 'relu4_2', 'layer for examine')
cmd:option('-seed', 876)
cmd:option('-output_image', 'out.png')

local params = cmd:parse(arg)

if params.seed >= 0 then
  torch.manualSeed(params.seed)
end

local content_image = image.load(params.content_image, 3)
content_image = image.scale(content_image, params.image_size, 'bilinear')
local content_image_caffe = preprocess(content_image):float()
local img = content_image_caffe:clone():float()


local cnn = loadcaffe.load(params.proto_file, params.model_file, "nn"):float()

local net = nn.Sequential()


for i = 1, #cnn do
      local layer = cnn:get(i)
      local typ = torch.type(layer)
      local name = layer.name
      print(name, typ)
      net:add(layer)
      if (name == params.layer) then break end
      if (i == #cnn) then 
        print("No such layer: "..params.layer)
        return 
      end   
end

local fmaps = net:forward(img)

local y = torch.sum(fmaps, 1)
local m = y:max()
y = y:mul(255):div(m)

local y3 = torch.Tensor(3,y:size(2),y:size(3))
local y1 = y[1]
y3[1] = y1
y3[2] = y1
y3[3] = y1
local disp = deprocess(y3:double())
disp = image.minmax{tensor=disp, min=0, max=1}
disp = image.scale(disp, content_image:size(3), content_image:size(2))
image.save(params.output_image, disp)
print("saving image ",params.output_image)