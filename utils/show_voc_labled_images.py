import cv2 as cv
import os
import xml.etree.cElementTree as ET


def show_voc(images_path,annotations_path,scale_ratio=1.0):
    xml_list = os.listdir(annotations_path)
    for file in xml_list:
        xml_file = os.path.join(annotations_path,file)
        tree = ET.ElementTree(file=xml_file)  
        root = tree.getroot()
        file_name = root.findtext('filename')
        pic_file = os.path.join(images_path,file_name)

        image_size = root.find('size')
        image_width = int(image_size.findtext('width'))
        image_height = int(image_size.findtext('height'))
        image_channels = int(image_size.findtext('depth'))
        rois = root.findall('object')
        bboxs = []
        roi_names = []
        for roi in rois:
            roi_name = roi.findtext('name')
            bbox = roi.find('bndbox')    
            xmin = int(bbox.findtext('xmin'))
            ymin = int(bbox.findtext('ymin'))       
            xmax = int(bbox.findtext('xmax'))
            ymax = int(bbox.findtext('ymax')) 
            box=[xmin,ymin,xmax,ymax]
            bboxs.append(box)
            roi_names.append(roi_name)
        if not os.path.isfile(pic_file):
            continue
        image = cv.imread(pic_file)
        if image is None:
            print('cannot open the picture')

        for idx in range(len(bboxs)):
            cv.rectangle(image,(bboxs[idx][0],bboxs[idx][1]),(bboxs[idx][2],bboxs[idx][3]),(0,0,255),2)
            cv.putText(image,roi_names[idx],(bboxs[idx][0],bboxs[idx][1]),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        rw = int(image_width*scale_ratio)
        rh = int(image_height*scale_ratio)
        rimg = cv.resize(image,(rw,rh))
        cv.imshow("image",rimg)
        k = cv.waitKey(0)&0xff
        if k == ord('c'):
            os.remove(xml_file)
            os.remove(pic_file)


if __name__ == "__main__":
    #当前文件路径
    #filepath = os.path.realpath(__file__)
    #print(filepath)
    #当前文件所在的目录，即父路径
    #parent_path = os.path.split(os.path.realpath(__file__))[0]
    parent_path = "D:\\labeled_inner\\aug"
    print(parent_path)
    
    images_path=os.path.join(parent_path,"JPEGImages")
    annotations_path=os.path.join(parent_path,"Annotations")
    #images_path = "D:\\szp_samples\\inside"
    #annotations_path = "D:\\szp_samples\\label_result"
    show_voc(images_path,annotations_path,0.5)
