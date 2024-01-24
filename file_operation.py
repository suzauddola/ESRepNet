import pandas as pd
import os


# os.chdir()
# rename
def _rename_inside_txt_file_list(file_with_names, new_file_location, server_location_image_name):
    new_file_to_save = open(new_file_location, "w")

    for index, image in file_with_names.iterrows():
        image_name = image['name']
        print(str(index) + "-->" + image_name)
        text = server_location_image_name + image_name + '.jpg' + "\n"
        new_file_to_save.write(text)
        # if index >= 4800:
        #     print(text)
        #     test_file.write(text)


# rename pest24 dataset list info .txt
def pest24_rename_train_test_val():
    value = "train"
    # value = "test"
    # value = "val"

    location_of_txt_file = r'F:/Datasets/Pest24/ImageSets/' + value + '.txt'
    load_file_with_name = pd.read_csv(location_of_txt_file, delim_whitespace=False, header=None, delimiter=',',
                                      names=['name'])

    new_file_location = r'F:/Datasets/Pest24/ImageSets_with_server_locations/' + value + '.txt'

    # server_location_image_name = r"/home/suza/yolo/yolov7/pest24/images/train/"
    server_location_image_name = r'./images/' + value + '/'

    _rename_inside_txt_file_list(load_file_with_name, new_file_location, server_location_image_name)


# rename TTOPv7 dataset list info .txt
def TTOPv7_rename_train_test_val():
    # value = "train"
    # value = "test"
    value = "val"

    location_of_txt_file = r'F:/Datasets/roboflow/TTOP-v7/total/imagesets/' + value + '.txt'
    load_file_with_name = pd.read_csv(location_of_txt_file, delim_whitespace=False, header=None, delimiter=',',
                                      names=['name'])

    new_file_location = r'F:/Datasets/roboflow/TTOP-v7/total/' + value + '.txt'

    # server_location_image_name = r"/home/suza/yolo/yolov7/pest24/images/train/"
    server_location_image_name = r'./images/' + value + '/'

    _rename_inside_txt_file_list(load_file_with_name, new_file_location, server_location_image_name)


# rad file from location and save,
def read_and_save(src, save_path, file_list):
    # read all the file name from the location
    # all_file = [file_name for file_name in os.listdir(src)]

    # # save list of file inside of text file
    # for single_file in all_file:
    #     file.write(single_file + '\n')
    #     file_list.append(single_file + '\n')

    # read all the file name from the location
    all_file = [file_name for file_name in os.listdir(src)]

    # dest_path = '/home/suza/YOLO/yolov7/TTOP_basic/train.txt'

    # save list of file inside of text file
    # with open(dest_path, 'w') as f_out:
    for single_file in all_file:
        save_path.write("./images/train/" + single_file + '\n')
        file_list.append(single_file + '\n')
            # f_out.write(single_file + '\n')

    # print(len(name_file))
    save_path.close()

    return file_list


# can read but can not save into the .txt file
def read_all_file_name_from_a_location_and_save_into_txt_file():
    # img_file_path = "D:/ResearchData/phd_research/Project/yolo/ScaledYOLOv4/coco/test2017"

    test_path = r"/home/suza/YOLO/yolov7/only_mosic/test/images"
    train_path = r"/home/suza/YOLO/yolov7/only_mosic/train/images"
    val_path = r"/home/suza/YOLO/yolov7/only_mosic/valid/images"

    img_file_path = val_path
    save_file_path = r"/home/suza/YOLO/yolov7/only_mosic/"
    file_list = []
    file_to_save = open(save_file_path + "val.txt", "w")
    files = read_and_save(img_file_path, file_to_save, file_list)
    # print("Done!")
    return files


if __name__ == '__main__':
    #
    # pest24_rename_train_test_val()
    #
    # # Read file from specifics location
    files = read_all_file_name_from_a_location_and_save_into_txt_file()
    # print(files)

    # TTOPv7_rename_train_test_val()

    print("Done!")