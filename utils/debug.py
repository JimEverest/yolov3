#██████╗░███████╗██████╗░██╗░░░██╗░██████╗░
#██╔══██╗██╔════╝██╔══██╗██║░░░██║██╔════╝░
#██║░░██║█████╗░░██████╦╝██║░░░██║██║░░██╗░
#██║░░██║██╔══╝░░██╔══██╗██║░░░██║██║░░╚██╗
#██████╔╝███████╗██████╦╝╚██████╔╝╚██████╔╝
#╚═════╝░╚══════╝╚═════╝░░╚═════╝░░╚═════╝░
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import os
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()



def save_image(tensor, id):
    dir = 'debug'
    try:
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    except:
        image = tensor
    #image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save('debug/s{}.jpg'.format(id))


def save_mosaic_img(imgs,targets,id):
    print("debug on the fly...")
    dir = 'debug'
    try:
        image = imgs.cpu().clone()  # we clone the tensor to not do changes on it
    except:
        print("cpu clone skipped...")
        image = imgs

    image = unloader(image)
    draw = ImageDraw.Draw(image)
    for tar in targets: # CLS,X1,Y1,X2,Y2
        # tar[2:4] <--------  CENTER
        try:
            x1,y1 = tar[0][1:3]
            print("x1,y1--->",(x1,y1)) 
            x2,y2 = tar[0][3:]
            print("x2,y2--->",(x2,y2))
        except:
            x1,y1 = tar[1:3]
            print("x1,y1--->",(x1,y1)) 
            x2,y2 = tar[3:]
            print("x2,y2--->",(x2,y2))

        draw.rectangle(((x1,y1), (x2,y2)), outline="#ff8888")
    # image.show()
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save('debug/mosaic_{}.jpg'.format(id))



def save_target_imgs(imgs,targets,id):
    dir = 'debug'
    imgs = imgs.cpu().clone()  
    print("Img count: ", str(imgs.shape[0]))
    for i,img in enumerate(imgs):
        print("------------------------- ",str(i)," ------------------------")
        select_tarIds = torch.where(targets[...,0]==i)
        select_tars = targets[select_tarIds]
        image = unloader(img)
        draw = ImageDraw.Draw(image)
        for tar in select_tars:
            # tar[2:4] <--------  CENTER 
            x1,y1 = (tar[2:4]*416 - tar[4:]*416/2).cpu().numpy() 
            print("x1,y1--->",(x1,y1)) 
            x2,y2 = (tar[2:4]*416 + tar[4:]*416/2).cpu().numpy()
            print("x2,y2--->",(x2,y2))
            
            draw.rectangle(((x1,y1), (x2,y2)), outline="#ff8888")
        # image.show()
        if not os.path.exists(dir):
            os.makedirs(dir)
        image.save('debug/s_{}_{}.jpg'.format(id,i))
        print("--------------------------------------------------")


    
def load_img_labels(img_path,xywh,id):
    dir = 'debug'
    if(type(img_path) == str):
        if(img_path.endswith(".txt")):
            img_path = img_path.replace("labels/","images/").replace(".txt",".jpg")
        image = Image.open(img_path)
    else:
        image = img_path
    draw = ImageDraw.Draw(image)
    tar=xywh
    img_w = image.size[0]
    img_h = image.size[1]

    # tar[0:2] <--------  CENTER 
    x1= (tar[0]*img_w - tar[2]*img_w/2)
    y1= (tar[1]*img_h - tar[3]*img_h/2)
    print("x1,y1--->",(x1,y1)) 
    x2= (tar[0]*img_w + tar[2]*img_w/2)
    y2= (tar[1]*img_h + tar[3]*img_h/2)
    print("x2,y2--->",(x2,y2))
    
    draw.rectangle(((x1,y1), (x2,y2)), outline="#ff8888")
    image.show()
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # image.save('debug/s_{}_{}.jpg'.format(id,i))
    # print("--------------------------------------------------")





