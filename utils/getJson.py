import requests
import argparse
import json 
import os

def getData(url, out_path, names = None,length_figure=None):
    baseUrl = "https://openi.nlm.nih.gov"
    numFigure = 200
    max_count = 100
    for i in range(len(names)): 
        print(f'Downloading {names[i]}')
        print('=' * 200)
        total = length_figure[i] // max_count
        start = 0
        end = max_count
        images_info = []
        for j in range(total):
            params = {
                "m": str(start + j*max_count),
                "n": str(end + j*max_count),
                "query": names[i],
                "it":"xm"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
        # Request was successful
                data = response.json()
                for item in data['list']:
                    images_info.append(item)
                print(len(images_info))
        # Save to JSON file
            else:
                print("Error:", response.status_code)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        with open(os.path.join(out_path,names[i]+"image_data.json"), 'w') as f:
            json.dump(images_info, f, indent=4)
            print(f"Metadata for {len(images_info)} images saved successfully.")
        print('=' * 200)



def get_total_length(url,names=None):
    len = []
    baseUrl = "https://openi.nlm.nih.gov"
    numFigure = 200
    for name in names: 
        params = {
        "m": "1",
        "n": str(200),
        "query": name,
        "it":"xm"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
        # Request was successful
            data = response.json()
        len.append(int(data['total']))
    return len

def get_total_length_by_article(url,article_names = None):
    len = []
    baseUrl = "https://openi.nlm.nih.gov"
    numFigure = 200
    for name in article_names: 
        params = {
        "m": "1",
        "n": str(200),
        "coll": name,
        "it":"xm"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
        # Request was successful
            data = response.json()
        len.append(int(data['total']))
    return len


def get_data_by_article(url,out_path,article_names=None,length_figure=None):
    baseUrl = "https://openi.nlm.nih.gov"
    max_count = 100
    for i in range(len(article_names)): 
        print(f'Downloading {article_names[i]}')
        print('=' * 200)
        total = length_figure[i] // max_count
        start = 0
        end = max_count
        images_info = []
        for j in range(total):
            params = {
                "m": str(start + j*max_count),
                "n": str(end + j*max_count),
                "coll": article_names[i],
                "it":"xm"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
        # Request was successful
                data = response.json()
                print(data)
                for item in data['list']:
                    images_info.append(item)
                print(len(images_info))
        # Save to JSON file
            else:
                print("Error:", response.status_code)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        with open(os.path.join(out_path,article_names[i]+"image_data.json"), 'w') as f:
            json.dump(images_info, f, indent=4)
            print(f"Metadata for {len(images_info)} images saved successfully.")
        print('=' * 200)

def main(args):
    #names = ['clinical photograph','heatmap','X ray','pie plot','micrograph','line graph','scatterplot']
    article_names = ['pmc']
    url_path = "https://openi.nlm.nih.gov/api/search"
    #getData(url,args.save_path,names)
    #len = get_total_length(url,names)
    #getData(url,args.save_path,names=names,length_figure=len)
    #print(get_total_length_by_article(url,article_names))
    article_len = get_total_length_by_article(url_path,article_names)
    print(article_len)
    get_data_by_article(url_path,args.save_path,article_names,[1000])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--save_path', type=str, help='Path to the directory')
    args = parser.parse_args()
    main(args)

