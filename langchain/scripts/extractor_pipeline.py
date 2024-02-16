import cv2
import os
import glob
import pandas as pd
from pathlib import Path
import logging

from matching_detector import MatchingDetector

class ExtractorPipeline():
    """
    performs feature extraction from all image files in the assets folder

    Creates a directory where extracted features will be saved as CSV files

    Extracted features include: Logo, CTA button, engagement button, objects, facial features, dominant colours, texts

    Parameters: full path to the assets folder
    """

    def __init__(self, data_folder) -> None:
        self.assets_folder = data_folder
        assets_dir_path = os.path.dirname(self.assets_folder)
        self.CWD = os.getcwd()
        self.extracted_path = str(
            Path(assets_dir_path).parent)+"/extracted_features"

        if not os.path.isdir(self.extracted_path):
            os.makedirs(self.extracted_path)
        
    def segment_extractor(self, segment_name):
        """
        extract the location of the logo from all preview images in the assets folder
        """
        folder_list = glob.glob(self.assets_folder)

        print(len(folder_list))

        t_matching = MatchingDetector('img')
        # store logo dimensions in a list
        logo_feature = []
        count = 0

        for folder in folder_list:
            # access folders in the assets directory
            query_img = os.path.join(folder, '_preview.png')
            train_img = os.path.join(folder, f'{segment_name}.png')


            # check if files exist
            if os.path.exists(query_img) and os.path.exists(train_img):
                count += 1

                location, bottom_right, top_left, res, img = t_matching.template_matching_image(
                    train_img, query_img, method=cv2.TM_CCOEFF_NORMED)

                if (bottom_right is not None) and (location is not None) and (top_left is not None):
                    logo_feature.append([folder.split('/')[-1], location[0], location[1],
                                        bottom_right[0], bottom_right[1], top_left[0], top_left[1]])
                else:
                    logo_feature.append(
                        [folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
            else:
                # if image does not exist
                logo_feature.append([folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
        
        print(f"count: {count}")

        # save the list elements as a dataframe
        df = pd.DataFrame(logo_feature, columns=[
            'id', f'{segment_name}_w', f'{segment_name}_h', f'{segment_name}_btrx', f'{segment_name}_btry', f'{segment_name}_tltx', f'{segment_name}_tlty'])

        # save dataframe as csv file
        df.to_csv(f'{self.extracted_path}/{segment_name}_positions.csv', index=False)

        return df


    def logo_extractor(self):
        """
        extract the location of the logo from all preview images in the assets folder
        """
        folder_list = glob.glob(self.assets_folder)

        print(len(folder_list))

        t_matching = MatchingDetector('img')
        # store logo dimensions in a list
        logo_feature = []
        count = 0

        for folder in folder_list:
            # access folders in the assets directory
            query_img = os.path.join(folder, '_preview.png')
            train_img = os.path.join(folder, 'logo.png')


            # check if files exist
            if os.path.exists(query_img) and os.path.exists(train_img):
                count += 1

                location, bottom_right, top_left, res, img = t_matching.template_matching_image(
                    train_img, query_img, method=cv2.TM_CCOEFF_NORMED)

                if (bottom_right is not None) and (location is not None) and (top_left is not None):
                    logo_feature.append([folder.split('/')[-1], location[0], location[1],
                                        bottom_right[0], bottom_right[1], top_left[0], top_left[1]])
                else:
                    logo_feature.append(
                        [folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
            else:
                # if image does not exist
                logo_feature.append([folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
        
        print(f"count: {count}")

        # save the list elements as a dataframe
        df = pd.DataFrame(logo_feature, columns=[
            'id', 'logo_w', 'logo_h', 'logo_btrx', 'logo_btry', 'logo_tltx', 'logo_tlty'])

        # save dataframe as csv file
        df.to_csv(self.extracted_path+'/logo_positions.csv', index=False)

    def engagement_button(self):
        """
        extract the position of the engagement button from all preview images in the assets folder
        """
        # access assets directory
        folder_list = glob.glob(self.assets_folder)

        t_matching = MatchingDetector('img')

        # list to store engagement locations
        engagement_buttons = []
        for folder in folder_list:
            query_img = os.path.join(folder, '_preview.png')
            train_img = os.path.join(folder, 'engagement_instruction.png')
            # check if files exists
            if os.path.exists(query_img) and os.path.exists(train_img):
                location, bottom_right, top_left, res, img = t_matching.template_matching_image(
                    train_img, query_img, method=cv2.TM_CCOEFF_NORMED)
                if (bottom_right is not None) and (location is not None) and (top_left is not None):
                    engagement_buttons.append([folder.split(
                        '/')[-1], location[0], location[1], bottom_right[0], bottom_right[1], top_left[0], top_left[1]])
                else:
                    # if button is not present
                    engagement_buttons.append(
                        [folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
            else:
                # if path does not exist
                engagement_buttons.append(
                    [folder.split('/')[-1], 0, 0, 0, 0, 0, 0])

        # create dataframe
        df = pd.DataFrame(engagement_buttons, columns=[
                          'id', 'engagement_w', 'engagement_h', 'engagement_btrx', 'engagement_btry', 'engagement_tltx', 'engagement_tlty'])
        # save as CSV file
        df.to_csv(self.extracted_path +
                  '/engagement_txt_positions.csv', index=False)

    def get_CTA_positions(self):
        """
        extract the position of the engagement button from all preview images in the assets folder
        """
        # access assets directory
        folder_list = glob.glob(self.assets_folder)

        t_matching = MatchingDetector('img')

        # create list to store CTA positions
        cta_positions = []
        for folder in folder_list:
            query_img = os.path.join(folder, '_preview.png')
            train_img = os.path.join(folder, 'cta.png')
            # check if files exist
            if os.path.exists(query_img) and os.path.exists(train_img):
                location, bottom_right, top_left, res, img = t_matching.template_matching_image(
                    train_img, query_img, method=cv2.TM_CCOEFF_NORMED)
                if (bottom_right is not None) and (location is not None) and (top_left is not None):
                    cta_positions.append([folder.split(
                        '/')[-1], location[0], location[1], bottom_right[0], bottom_right[1], top_left[0], top_left[1]])
                else:
                    # CTA location not found
                    cta_positions.append(
                        [folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
            else:
                # files not found
                cta_positions.append([folder.split('/')[-1], 0, 0, 0, 0, 0, 0])
            # save as dataframe
            df = pd.DataFrame(cta_positions, columns=[
                              'id', 'cta_w', 'cta_h', 'cta_btrx', 'cta_btry', 'cta_tltx', 'cta_tlty'])
            # save as CSV file
            df.to_csv(self.extracted_path+'/cta_txt_position.csv', index=False)

    # def detect_objects(self):
    #     """
    #     extract the position of objects from all preview images in the assets folder
    #     """
    #     od = ObjectDetector('img')
    #     net = od.load_yolo
    #     # access assets directory
    #     folder_list = glob.glob(self.assets_folder)
    #     object_id_feature = [0]*100
    #     right_btmx = [0]*100
    #     right_btmy = [0]*100
    #     top_lftx = [0]*100
    #     top_lfty = [0]*100

    #     # create list to store objects present in the image
    #     object_features = []
    #     for folder in folder_list:
    #         query_img = os.path.join(folder, '_preview.png')
    #         # check if file exists
    #         if os.path.exists(query_img):
    #             class_IDs, scores, bounding_boxs, img = od. detect_from_image(
    #                 query_img, net)
    #             id_list = class_IDs.squeeze().asnumpy().tolist()
    #             for i in range(len(id_list)):
    #                 if id_list[i] != -1:
    #                     object_id_feature[int(id_list[i])] = 1
    #                     right_btmx[int(id_list[i])] = bounding_boxs[0][i][0]
    #                     right_btmy[int(id_list[i])] = bounding_boxs[0][i][1]
    #                     top_lftx[int(id_list[i])] = bounding_boxs[0][i][2]
    #                     top_lfty[int(id_list[i])] = bounding_boxs[0][i][3]
    #                     # add object position to list
    #                     object_features.append([folder.split(
    #                         '/')[-1]] + object_id_feature + right_btmx + right_btmy + top_lftx + top_lfty)
    #                 else:
    #                     # object not detected
    #                     object_features.append([folder.split('/')[-1]]+[0]*500)
    #             # save features list as dataframe
    #             df = pd.DataFrame(object_features)
    #             # save as CSV file
    #             df.to_csv(self.extracted_path +
    #                       '/objects_detected.csv', index=False)

    # def detect_facial_features(self):
    #     """
    #     extract dominant facial features from all preview images in the assets folder
    #     """
    #     # access assets directory
    #     folder_list = glob.glob(self.assets_folder)
    #     # create list to store the results of facial analyses
    #     dominant_features = []
    #     # instantiate face detection class
    #     detection_class = FaceDetection()

    #     for folder in folder_list:
    #         query_img = os.path.join(folder, '_preview.png')
    #         # check if files exist
    #         if os.path.exists(query_img):
    #             # detect faces
    #             face_detected = detection_class.detect_faces(query_img)
    #             # if more than one face is detected
    #             if len(face_detected) > 0:
    #                 # face detected -> perform analysis
    #                 face_analysis = detection_class.get_face_analysis(
    #                     query_img)
    #                 # store analysis result in list
    #                 dominant_race = face_analysis['dominant_race']
    #                 dominant_emotion = face_analysis['dominant_emotion']
    #                 age = face_analysis['age']
    #                 gender = face_analysis['gender']
    #                 face_location = face_analysis['region']
    #                 dominant_features.append([folder.split('/')[-1], dominant_race, dominant_emotion, age,
    #                                          gender, face_location['x'], face_location['y'], face_location['w'], face_location['h']])
    #             else:
    #                 # no faces were detected
    #                 dominant_features.append(
    #                     [folder.split('/')[-1], 0, 0, 0, 0, 0, 0, 0, 0])
    #         else:
    #             # file not found
    #             dominant_features.append(
    #                 [folder.split('/')[-1], 0, 0, 0, 0, 0, 0, 0, 0])

    #         df = pd.DataFrame(dominant_features, columns=[
    #                           'id', 'race', 'emotion', 'age', 'gender', 'face_location_x', 'face_location_y', 'face_width', 'face_height'])
    #         # save as CSV file
    #         df.to_csv(self.extracted_path+'/facial_features.csv', index=False)

   