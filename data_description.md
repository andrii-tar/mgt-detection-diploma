## Data directory structure

## data_base_dir/

- ## raw/    
contains raw downloaded datasets   

    - dataset_1_downloaded
    - dataset_2_downloaded
    - dataset_3_downloaded
    - dataset_4_generated
    - ...
 --- 
- ## extracted/  
contains extracted texts for processing.  
texts are with specific source, length, type of origin  
MG - machine generated  
HW - human written  
data set name used in experiments: {dataset_name}__{modification}

    - dataset_1__full  
        - MG
            - texts.csv
        - HW
            - texts.csv
    - dataset_1__first_10_sent
        - MG
        - HW
    - dataset_1__first_paragraph ...
    - dataset_2__full ...
    --- 
- ## processed  
data with calculated logprobs

    - {dataset_name}__{modification}
        - HW
            - data.csv
            - metrics.csv
        - MG
            - data.csv
            - metrics.csv

    - ...

## dataset_mapping.csv

    dataset_full_name: {dataset_name}__{modification}
    dataset_id: int
    dataset_label: short string for visualization plots
    
    

## texts.csv :
extracted texts

    - id
    - document_id
    - raw_text_id
    - start_pos
    - end_pos
    - text

## processed data {HW/MG}
human written and machine generated data files have same structure
data origin will be marked on data merging

    ## data.csv
    - id
    - dataset_id
    - text_id
    - logprobs_100
    - logprobs_20  

    ## metrics.csv
    - id
    - dataset_id 
    - text_id
    - perplexity_100 (float)
    - d_score_100 (float) 
    - perplexity_20 (float)
    - d_score_20  (float)
    ...
