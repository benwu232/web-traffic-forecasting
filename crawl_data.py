import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import re
import urllib
import requests
import datetime
from mwviews.api import PageviewsClient


from lib.foundation import *



def get_page_info(raw_data_file, load_len=1000):
    train_data1 = load_data(raw_data_file, load_len=load_len)
    _, page_info = pre_process(train_data1)
    return page_info.values

def parse_full_name(pi):
    results = re.finditer(r'_', pi)
    positions_ = []
    for r in results:
        positions_.append(r.start())

    agent = (pi[positions_[-1]+1:])
    access = (pi[positions_[-2]+1:positions_[-1]])
    cat = (pi[positions_[-3]+1:positions_[-2]])
    name = (pi[:positions_[-3]])

    return name, cat, access, agent

NAME = 0
CAT = 1
ACCESS = 2
AGENT = 3
def parse_page_info(page_info):
    pi_detail = np.empty((len(page_info), 4), dtype=object)

    for k, pi in enumerate(page_info):
        results = re.finditer(r'_', pi)
        positions_ = []
        for r in results:
            positions_.append(r.start())

        pi_detail[k, AGENT] = (pi[positions_[-1]+1:])
        pi_detail[k, ACCESS] = (pi[positions_[-2]+1:positions_[-1]])
        pi_detail[k, CAT] = (pi[positions_[-3]+1:positions_[-2]])
        pi_detail[k, NAME] = (pi[:positions_[-3]])

    access_set = set(pi_detail[:, ACCESS])
    agent_set = set(pi_detail[:, AGENT])
    category_set = set(pi_detail[:, CAT])
    print('category', len(category_set), category_set)
    print('access', len(access_set), access_set)
    print('agent', len(agent_set), agent_set)

    return pi_detail


def parse_page_info1(page_info):
    pinfo = {}
    pinfo['name'] = []
    pinfo['category'] = []
    pinfo['access'] = []
    pinfo['agent'] = []

    pattern = re.compile(r'_')
    for pi in page_info:
        results = pattern.finditer(pi)
        positions_ = []
        for r in results:
            positions_.append(r.start())

        pinfo['agent'].append(pi[positions_[-1]+1:])
        pinfo['access'].append(pi[positions_[-2]+1:positions_[-1]])
        pinfo['category'].append(pi[positions_[-3]+1:positions_[-2]])
        pinfo['name'].append(pi[:positions_[-3]])

    access_set = set(pinfo['access'])
    agent_set = set(pinfo['agent'])
    category_set = set(pinfo['category'])
    print('category', len(category_set), category_set)
    print('access', len(access_set), access_set)
    print('agent', len(agent_set), agent_set)

    return pinfo

def dt2str(dt):
    return dt.strftime("%Y-%m-%d")

def pv2dict(pv, access, agent, cat):
    dict_set = OrderedDict()
    none_cnt = {}
    surfix = '_{}_{}_{}'.format(cat, access, agent)
    for k, day_dt in enumerate(pv):
        day_str = dt2str(day_dt)
        for name in pv[day_dt]:
            full_name = name + surfix
            if full_name not in dict_set:
                dict_set[full_name] = OrderedDict()
                dict_set[full_name]['Page'] = full_name
                none_cnt[full_name] = 0
            dict_set[full_name][day_str] = pv[day_dt][name]
            if dict_set[full_name][day_str] is None:
                none_cnt[full_name] += 1

    #delete items with too many nones
    pv_len_thresh = len(pv) // 2
    for full_name in none_cnt:
        if none_cnt[full_name] > pv_len_thresh:
            dict_set.pop(full_name, None)

    #print('{} items downloaded.'.format(len(dict_set)))
    return dict_set

page_info_file = '../output/page_info.pkl'
def download_page_view(pi_detail, page_info, start ='2015070100', end='', batch_size = 10):
    pvc = PageviewsClient()
    if end == '':
        end = dt.datetime.now().strftime("%Y%m%d00")

    batch_dict = OrderedDict()

    for k in (range(0, len(pi_detail), batch_size)):
        batch_list = list(pi_detail[k:k + batch_size, NAME])
        print('Step', k)
        print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('Downloading: ', batch_list)
        for agent in ['spider', 'all-agents']:
            for access in access_type.keys():
                for cat in categories.keys():
                    try:
                        pv = pvc.article_views(cat, batch_list, granularity='daily', access=access, agent=agent, start=start, end=end)
                        dict_set = pv2dict(pv, access, agent, cat, page_info)
                        batch_dict.update(dict_set)
                    except:
                        pass
    return batch_dict

def download_pv_one(pars):
    batch_list = pars[0]
    name_list = pars[1]
    start = pars[2]
    end = pars[3]
    k = pars[4]

    pvc = PageviewsClient()
    batch_dict = OrderedDict()
    print('************************* Step {} ********************************'.format(k))
    print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print('Downloading: ', batch_list)
    for agent in ['spider', 'all-agents']:
        for access in access_type.keys():
            for cat in categories.keys():
                for name in batch_list:
                    full_name = '{}_{}_{}_{}'.format(name, cat, access, agent)
                    if full_name in name_list:
                        batch_list.remove(name)
                if batch_list:
                    try:
                        pv = pvc.article_views(cat, batch_list, granularity='daily', access=access, agent=agent, start=start, end=end)
                        dict_set = pv2dict(pv, access, agent, cat)
                        batch_dict.update(dict_set)
                    except:
                        pass
    return batch_dict

def start_process():
    print('Starting', multiprocessing.current_process().name)

def download_pv_mp(pi_detail, start ='2015070100', end='', batch_size=10, pool_size=10, section_size=10000):
    if end == '':
        end = dt.datetime.now().strftime("%Y%m%d00")

    if os.path.isfile(page_info_file):
        pi_dict = load_dump(page_info_file)
    else:
        pi_dict = OrderedDict()

    name_list = pi_dict.keys()

    for s in (range(0, len(pi_detail), section_size)):
        pars = []
        for k in (range(s+0, min(s+section_size, len(pi_detail)), batch_size)):
            batch_list = list(pi_detail[k:k + batch_size, NAME])
            pars.append((batch_list, name_list, start, end, k))

        pool = ThreadPool(processes=pool_size, initializer=start_process,)
        pool_results = pool.map(download_pv_one, pars)
        pool.close()
        pool.join()

        for result in pool_results:
            for key in result.keys():
                if key not in pi_dict:
                    pi_dict[key] = result[key]
                else:
                    pi_dict[key].update(result[key])

        save_dump(pi_dict, page_info_file)

def download_pv_mp1(pi_detail, start ='2015070100', end='', batch_size=10, pool_size=10, section_size=10000):
    if end == '':
        end = dt.datetime.now().strftime("%Y%m%d00")

    if os.path.isfile(page_info_file):
        pi_dict = load_dump(page_info_file)
        name_list = []
    else:
        pi_dict = OrderedDict()
        name_list = pi_dict.keys()

    for s in (range(0, len(pi_detail), section_size)):
        pars = []
        for k in (range(s+0, min(s+section_size, len(pi_detail)), batch_size)):
            batch_list = list(pi_detail[k:k + batch_size, NAME])
            pars.append((batch_list, name_list, start, end, k))

        pool = ThreadPool(processes=pool_size, initializer=start_process,)
        pool_results = pool.map(download_pv_one, pars)
        pool.close()
        pool.join()

        for result in pool_results:
            for key in result.keys():
                if key not in pi_dict:
                    pi_dict[key] = result[key]
                else:
                    pi_dict[key].update(result[key])

        save_dump(pi_dict, page_info_file)


def download_page_view1(pi_detail, page_info, start ='2015070100', end='', batch_size = 10):
    pvc = PageviewsClient()
    if end == '':
        end = dt.datetime.now().strftime("%Y%m%d00")

    if os.path.isfile(page_info_file):
        pi_dict = load_dump(page_info_file)
    else:
        pi_dict=OrderedDict()

    for k in (range(0, len(pi_detail), batch_size)):
        batch_list = list(pi_detail[k:k + batch_size, NAME])
        print('Step', k)
        print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('Downloading: ', batch_list)
        for agent in ['spider', 'all-agents']:
            for access in access_type.keys():
                for cat in categories.keys():
                    try:
                        pv = pvc.article_views(cat, batch_list, granularity='daily', access=access, agent=agent, start=start, end=end)
                        dict_set = pv2dict(pv, access, agent, cat, page_info)
                        pi_dict.update(dict_set)
                    except:
                        pass
    save_dump(pi_dict, page_info_file)

def dict2df(pv_dict):
    df = pd.DataFrame(pv_dict).T
    df = df.reset_index()
    df = df.drop('Page', axis=1)
    df = df.rename(columns={'index': 'Page'})
    return df



if __name__ == '__main__':
    '''
    pvc = PageviewsClient()
    pv = pvc.article_views('en.wikipedia.org', ['刘亦菲'], granularity='daily', access='all-access', agent='user', start='2015090100', end='2017091000')
    pvc.article_views('en.wikipedia', ['Selfie', 'Cat'], granularity='monthly', start='2016020100', end='2016043000')
    pvc.article_views('es.wikipedia', ['Resident_Evil:_Capítulo_Final'], granularity='daily', access='all-access', agent='user', start='2017090100', end='2017091000')
    pvc.article_views('zh.wikipedia.org', ['AlphaGo'], granularity='daily', access='all-access', agent='user', start='2015090100', end='2017091000')
    '''

    raw_data_file = '../input/train_2.csv'
    load_len = 3
    load_len = 1023
    load_len = 93
    load_len = 27
    load_len = -1

    page_info = get_page_info(raw_data_file, load_len)
    pi_detail = parse_page_info(page_info)
    #download_page_view(pi_detail, page_info, batch_size=10)
    #page_info = np.flip(page_info, 0)
    download_pv_mp(pi_detail, start='2015070100', end='', batch_size=1000, pool_size=3, section_size=2000)
