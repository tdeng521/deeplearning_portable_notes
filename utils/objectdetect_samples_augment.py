'''
安装依赖
pip install six numpy scipy Pillow matplotlib scikit-image imageio Shapely -i https://pypi.tuna.tsinghua.edu.cn/simple
安装imgaug
pip install imgaug
'''

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import os
import numpy as np
import xml.etree.cElementTree as ET
from lxml import etree
from PIL import Image

class VocAnnotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
 
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
 
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
 
        child3 = etree.SubElement(self.root, "source")
 
        child4 = etree.SubElement(child3, "annotation")
 
    def set_size(self,witdh,height,channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self,label,xmin,ymin,xmax,ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(int(xmin))
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(int(ymin))
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(int(xmax))
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(int(ymax))

def modify_rect(box,width,height):
    if box.x1<0:
        box.x1=0
    if box.y1<0:
        box.y1=0
    if box.x2>width:
        box.x2=width-1
    if box.y2>height:
        box.y2=height-1
    return box
if __name__ == "__main__":
    # filepath = os.path.realpath(__file__)
    # root_path = os.path.split(os.path.realpath(__file__))[0]
    # print(root_path)
    root_path = "D:\\labeled_inner\\aug"
    src_images_dir = "D:\\labeled_inner\\label_inner\\labeled\\JPEGImages"
    src_annotations_dir = "D:\\labeled_inner\\label_inner\\labeled\\Annotations"
    #src_images_dir = os.path.join(root_path,'data\\voc\\JPEGImages')
    #src_annotations_dir = os.path.join(root_path,'data\\voc\\Annotations')
    aug_images_dir = os.path.join(root_path,'JPEGImages')
    aug_annotations_dir=os.path.join(root_path,'Annotations')
    sometimes = lambda aug: iaa.Sometimes(p=0.5, then_list=aug)

    seq = iaa.Sequential([
        iaa.Affine(rotate=(-30, 30),scale=(0.7,1.3)),
        iaa.AdditiveGaussianNoise(scale=(5, 15)),
        iaa.Fliplr(0.5),
        # sometimes(iaa.Affine(                          #对一部分图像做仿射变换
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  #图像缩放为80%到120%之间
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
        #     rotate=(-30, 30),   #旋转±30度之间
        #     shear=(-16, 16),    #剪切变换±16度，（矩形变平行四边形）
        #     order=[0, 1],   #使用最邻近差值或者双线性差值
        #     cval=(0, 255),  #全白全黑填充
        #     mode=ia.ALL    #定义填充图像外区域的方法
        # )),
        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf((1, 5),
            [
                # 将部分图像进行超像素的表示。
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),
                #用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                    iaa.MedianBlur(k=(3, 7)),
                ]),
                #锐化处理
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                #边缘检测，将检测到的赋值0或者255然后叠在原图上
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
                # 加入高斯噪声
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
                # 将1%到5%的像素设置为黑色
			    # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.10), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
                #5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                # 每个像素随机加减-10到10之间的数
                iaa.Add((-10, 10), per_channel=0.5),
                # 像素乘上0.5或者1.5之间的数字.
                iaa.Multiply((0.8, 1.2), per_channel=0.5),

                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                # 将RGB变成灰度图然后乘alpha加在原图上
                iaa.Grayscale(alpha=(0.0, 0.2)),
                #把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                # sometimes(
                #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                # ),
                # 扭曲图像的局部区域
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                iaa.GammaContrast((0.5,1.5)),
                iaa.FastSnowyLandscape(lightness_threshold=(100, 200),lightness_multiplier=(0.5, 1.5)),
                iaa.Snowflakes(),
                iaa.AddToSaturation((-50,50))
            ],
            random_order=True # 随机的顺序把这些操作用在图像上
        )
    ])
    #seq = iaa.Grayscale(alpha=(0.1, 0.4))#以0.1-0.4的概率对图像进行灰度化
    #iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    #seq = iaa.GammaContrast((0.5,2.0))
    #seq = iaa.FastSnowyLandscape(lightness_threshold=(100, 255),lightness_multiplier=(0.5, 1.5))
    #seq = iaa.Fog()
    #seq = iaa.Snowflakes()
    #seq = iaa.Clouds()
    #seq = iaa.AddToSaturation((-50,50))
    seq_det = seq.to_deterministic()
    image_list = os.listdir(src_images_dir)

    count = 1
    for img_filename in image_list:#遍历所有图像
        image_file = os.path.join(src_images_dir,img_filename)
        xml_filename = os.path.splitext(img_filename)[0] + '.xml'
        xml_file = os.path.join(src_annotations_dir,xml_filename)
        if not os.path.isfile(xml_file):
            print(xml_file,'is not exist!')
            continue

        #解析标注文件
        xmltree = ET.ElementTree(file=xml_file)  
        root = xmltree.getroot()
        filename = root.findtext('filename')
        image_size = root.find('size')
        image_width = int(image_size.findtext('width'))
        image_height = int(image_size.findtext('height'))
        image_channels = int(image_size.findtext('depth'))
        rois_bboxs=[]   #roi标注框
        rois_labels=[]  #roi label
        rois = root.findall('object')
        for roi in rois:
            roi_label = roi.findtext('name')
            bbox = roi.find('bndbox')    
            xmin = int(bbox.findtext('xmin'))
            ymin = int(bbox.findtext('ymin'))       
            xmax = int(bbox.findtext('xmax'))
            ymax = int(bbox.findtext('ymax')) 
            box=[xmin,ymin,xmax,ymax]
            rois_bboxs.append(box)
            rois_labels.append(roi_label)
        
        im = Image.open(image_file)
        img = np.asarray(im)
        img_h,img_w,img_depth=img.shape

        temp_aug_bbox = []
        boxes_img_aug_list=[]#变化后的box
        for bbox in rois_bboxs:
            temp_aug_bbox.append(ia.BoundingBox(x1=bbox[0], 
                                            x2=bbox[2], 
                                            y1=bbox[1], 
                                            y2=bbox[3]))
        bbs = ia.BoundingBoxesOnImage(temp_aug_bbox, shape=img.shape)

        for _ in range(2):
            seq_det = seq.to_deterministic()
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            image_aug = seq_det.augment_image(img)
            im = Image.fromarray(image_aug)
            
            aug_image_file = os.path.join(aug_images_dir,str(count)+ 'aug.jpg')#aug后的图像
            aug_xml_file = os.path.join(aug_annotations_dir,str(count)+ 'aug.xml')#aug后的xml
            
            anno= VocAnnotations(str(count)+ 'aug.jpg')
            anno.set_size(img_w,img_h,img_depth)
            for i in range(len(bbs_aug.bounding_boxes)):
                roi_label = rois_labels[i]
                box = bbs_aug.bounding_boxes[i]
                box = modify_rect(box,image_width,image_height)
                anno.add_pic_attr(roi_label,box.x1,box.y1,box.x2,box.y2)

            anno.savefile(aug_xml_file)
            im.save(aug_image_file,format="JPEG")
            
            count += 1
            #im.show()
        #ia.imshow(np.hstack(image_aug))

        


        


        