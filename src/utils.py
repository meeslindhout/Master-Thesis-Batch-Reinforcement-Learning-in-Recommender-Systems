import pandas as pd
import os

def evaluation_results_to_csv(output,
                              dataset_name, 
                              algo_name, 
                              reward_func):
    """
    Converts evaluation results to a CSV file and stores it in evaluation_results folder.
    Column structure of CSV file:
    algo_name - reward_func - position - MRR - HIT 

    Args:
        output (list): The evaluation results of the evaluation function.
        algo_name (str, optional): The name of the algorithm. Defaults to 'VSTAN (Baseline)'.
        reward_func (str, optional): The reward function. Defaults to 'click=3, purchase=10'.

    """
    
    if output is None:
        raise ValueError('Output is None')
    
    df = pd.DataFrame(output, columns=['Metric', 'Value', 'Bin', 'Pos'])
    df = df.drop(columns=['Bin', 'Pos'])
    df['Dataset'] = dataset_name
    df['Algorithm'] = algo_name
    df['Reward Function'] = reward_func
    
    # Reorder columns
    df = df[['Dataset', 'Algorithm', 'Reward Function', 'Metric', 'Value']]
   
    # create a folder to store the evaluation results
    os.makedirs('evaluation results', exist_ok=True)
    
    # write the dataframes to csv
    df.to_csv(f'evaluation results/{dataset_name} - {algo_name} ({reward_func}).csv', index=False)
    
    display(df)