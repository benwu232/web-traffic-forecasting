import sys, os
import yaml
import pandas as pd
import zipfile
import sklearn
import tqdm


def fusion1(fusion_file_list, submission_csv):
    print('Fusioning ...')
    base_dir = '../output'
    df_list = []
    for k, file_name in enumerate(fusion_file_list):
        fusion_file = os.path.join(base_dir, file_name)
        df_list.append(pd.read_csv(fusion_file))
        df_list[k] = sklearn.utils.shuffle(df_list[k])
        #df_list[k] = df_list[k].sort_values('Id')

    df_sum = df_list[0]
    for k in range(1, len(df_list)):
        df_sum['Visits'] += df_list[k]['Visits']
    df_sum['Visits'] /= len(df_list)

    print('Sorting ...')
    df_sum = df_sum.sort_values('Id')
    #df_sum = df_sum.sort_values('Visits')
    df_sum.to_csv(submission_csv, index=False, float_format='%.3f')

    '''
    print('Compressing ...')
    submission_file = submission_csv[:-3] + 'zip'
    zout = zipfile.ZipFile(submission_file, "w", zipfile.ZIP_DEFLATED)
    print('Compress {} ...'.format(submission_csv))
    zout.write(submission_csv, arcname=os.path.basename(submission_csv))
    zout.close()
    os.remove(submission_csv)
    '''
    print('Bingo!')


def fusion(fusion_file_list, submission_csv):
    print('Fusioning ...')
    base_dir = '../output'
    df_sum = pd.read_csv('../input/sample_submission_2.csv')
    df_sum = df_sum.sort_values('Id')
    for k, file_name in enumerate(fusion_file_list):
        fusion_file = os.path.join(base_dir, file_name)
        df_one = (pd.read_csv(fusion_file))
        print('Sorting ...', fusion_file)
        df_one = df_one.sort_values('Id')

        df_sum['Visits'] = df_sum['Visits'].values + df_one['Visits'].values
        '''
        continue

        for k in range(len(df_sum)):
            if k % 1000 == 0:
                print(k)
            if df_sum.iloc[k]['Id'] != df_one.iloc[k]['Id']:
                print('Wrong at {}, sum_id: {}, one_id: {}'.format(k, df_sum.iloc[k]['Id'], df_one.iloc[k]['Id']))
            df_sum_copy.set_value(k, 'Visits', df_sum.iloc[k]['Visits'] + df_one.iloc[k]['Visits'])
            df_sum_copy.set_value(k, 'Id', df_sum.iloc[k]['Id'])
        print(df_sum_copy.iloc[:10])
        exit()
        df_sum = df_sum_copy.sort_values('Id')
        df_sum_copy = df_sum.copy()
        #df_sum['Visits'] += df_one['Visits']
        '''

    df_sum['Visits'] /= len(fusion_file_list)

    #df_sum = df_sum.sort_values('Id')
    #df_sum = df_sum.sort_values('Visits')
    print('Writing to {}'.format(submission_csv))
    df_sum.to_csv(submission_csv, index=False, float_format='%.3f')

    print('Compressing ...')
    submission_file = submission_csv[:-3] + 'zip'
    zout = zipfile.ZipFile(submission_file, "w", zipfile.ZIP_DEFLATED)
    print('Compress {} ...'.format(submission_csv))
    zout.write(submission_csv, arcname=os.path.basename(submission_csv))
    zout.close()
    os.remove(submission_csv)

    print('Bingo!')



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Need configuration file name.')
        exit()

    config_file = sys.argv[1]
    with open('./yaml/' + config_file, 'r') as f:
        pars = yaml.safe_load(f)

    out_file = '../output/' + config_file[:-4] + 'csv'
    fusion(pars['fusion_files'], out_file)
