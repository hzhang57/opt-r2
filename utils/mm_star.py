import pandas as pd
from utils.data_utils import save_image_to_folder_base64


def load_open_compass(file_path):
    """
    Read TSV file using pandas (recommended for data analysis)
    """
    print("\n=== Reading TSV with pandas ===")
    try:
        # Read TSV file
        df = pd.read_csv(file_path, sep='\t')
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nBasic statistics:")
        print(df.describe())
        
        df_list = []
        for x, row in df.iterrows():
            #index= row['index']
            #print(index)
            df_list.append(row)
        return df_list
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    

class format_mmstar_dataset_oc:

    def __init__(self, dataset, output_file, output_images_folder):
        self.dataset = dataset
        self.output_file = output_file
        self.output_images_folder = output_images_folder


    def format_a_sample(self, a_sample):
        index    = a_sample["index"]
        question = a_sample["question"]
        question = question.replace('<image 1>', '')
        image    = a_sample["image"]
        quest_type = a_sample["category"]
        quest_type_l2 = a_sample["l2_category"]
        answer   = a_sample["answer"]
        opt_A = a_sample['A']
        opt_B = a_sample['B']
        opt_C = a_sample['C']
        opt_D = a_sample['D']
        

        opt2ans = {
            'A': opt_A,
            'B': opt_B,
            'C': opt_C,
            'D': opt_D,
                }



        #print("IMAGE {}".format(image))
        # save image into folder
        file_name = save_image_to_folder_base64(image, self.output_images_folder, index)

        ## pack all indexes, question, options, answer, quest_type into a new dict
        new_sample = {}
        new_sample['id'] = index
        new_sample['quest_type'] = quest_type
        new_sample['quest_type_l2'] = quest_type_l2
        new_sample['opt2ans'] = opt2ans
        #new_sample['all_choices'] = [ x for x in opt2ans.keys() ]
        new_sample['image'] = file_name
        new_sample['question'] = question
        new_sample['answer'] = answer
        new_sample['answer_str'] = "{}".format(opt2ans[answer])

        ###opt_str = ""
        ##for k in opt2ans.keys():
        ##    opt_str += "{}. {}\n".format(k, opt2ans[k])
        ## Add Prompt
        ##new_question = "<image>Question:\n{}\nOption: \n{}\nReturn only the option letter of the correct answer (e.g. A, B, C, D)".format(question, opt_str)

        ###new_question = "Question:\n{}\nOption:\n{}\nReturn only the option letter of the correct answer (e.g. A, B, C, D)".format(question, opt_str)
        ###gt_answer  = "{}. {}".format(answer, opt2ans[answer])
        ###new_sample['conversations'] = [
        ###    {
        ###        'from': "human",
        ###        'value': new_question
        ###    },
        ###    {
        ###        'from': "gpt",
        ###        'value': gt_answer
        ###    }
        ###]

        #print(new_sample)
        #print("**"*8)
        #print(new_question)
        #print(gt_answer)
        #print(new_sample)

        return new_sample

    def format(self):
        new_dataset = []
        n_vals = len(self.dataset)
        for ii in range(n_vals):
            a_sample = self.dataset[ii]
            a_new_sample = self.format_a_sample(a_sample)
            new_dataset.append(a_new_sample)
            #print(a_new_sample)

        ## save json
        return new_dataset

