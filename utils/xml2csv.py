"""
how to convert labeled samples from xml to csv
"""

import os, sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET
 
'''
batch convert xml to csv

Args:
     xml_ann_path:  xml标注文件的目录,
     csv_path   :   目标csv文件要保存的目录。一般在该目录下会生成train.csv和val.csv
     train_num_ratio:控制train.csv中样本所占的比例，比如总样本是100，如果train_num_ratio=0.9，则有90个xml文件中的样本进入到train.csv，其余的进入val.csv

'''
def xml_to_csv(xml_ann_path, csv_path,train_num_ratio = 0.9):
    xml_list_train = []
    xml_list_val = []
    xml_files = glob.glob(xml_ann_path + '/*.xml')
    xml_num = len(xml_files)#total xml num
    
    count = 0
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):

            #有时候可能遇到标注文件中的标注的名称并不喜欢，可以通过以下方式修改，修改完记得最后保存。
            if "Power_Cable" in member[0].text:
                rewrite = True
                member.find('name').text = "power_cable"
            if "Memory" in member[0].text:
                rewrite = True
                member.find('name').text = "memory"
            
            #
            try:
                #读取xml的内容
                '''
                <object>
                    <name>Power_Cable</name>
                    <bndbox>
                    <xmin>929</xmin>
                    <ymin>849</ymin>
                    <xmax>976</xmax>
                    <ymax>903</ymax>
                    </bndbox>
                </object>
                '''
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member.find('name').text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
            except Exception as e:
                    #读取xml的内容
                    '''
                    <object>
                        <name>Fan_Cable</name>
                        <pose>Unspecified</pose>
                        <truncated>0</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                            <xmin>364</xmin>
                            <ymin>539</ymin>
                            <xmax>405</xmax>
                            <ymax>584</ymax>
                        </bndbox>
                    </object>
                    '''
                    value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member.find('name').text,
                        int(member[1][0].text),
                        int(member[1][1].text),
                        int(member[1][2].text),
                        int(member[1][3].text)
                        )
            count = count + 1
            if count > xml_num*train_num_ratio:
                xml_list_val.append(value)
            else:
                xml_list_train.append(value)
        if rewrite:
            tree.write(xml_file)#重写xml文件
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_csv_file = os.path.join(csv_path,'train.csv')
    val_csv_file = os.path.join(csv_path,'val.csv')
    #生成train.csv
    xml_train_df = pd.DataFrame(xml_list_train, columns=column_name)
    xml_train_df.to_csv(train_csv_file, index=None)
    #生成val.csv
    xml_val_df = pd.DataFrame(xml_list_val, columns=column_name)
    xml_val_df.to_csv(val_csv_file, index=None)

    print('Successfully converted xml to csv.')
 
if __name__ == '__main__':
    # convert
    ann_xmls = "D:\\labeled_inner\\aug\\Annotations"
    outpath = "D:\\labeled_inner\\aug\\Annotations"
    xml_to_csv(ann_xmls, outpath)
