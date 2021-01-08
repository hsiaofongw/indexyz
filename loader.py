import os
import json
from regex import findall, sub
from math import remainder
from typing import List, Tuple

def print_found(name, article_len):
    print('Found: %s , length: %s' % (name, article_len, ))

def read_articles_from_folder(folder: str) -> Tuple[List[str], List[str]]:

    article_names = list()
    article_contents = list()

    sub_entries = os.listdir(folder)
    for sub_entry in sub_entries:
        this_path = os.path.join(os.path.abspath(folder), sub_entry)
        if os.path.isfile(this_path):

            print('opening ' + this_path)

            this_article_name = sub_entry
            this_article_content = str()
            
            with open(this_path, 'r') as f:
                this_article_content = f.read()
            
            print_found(this_article_name, len(this_article_content))
            
            article_names.append(this_article_name)
            article_contents.append(this_article_content)
        else:

            print('ommit ' + this_path)
    
    print('Found %s articles at total.' % (str(len(article_names)), ))

    return (
        article_names,
        article_contents,
    )

def read_articles_from_json(jsonPath: str) -> Tuple[List[str], List[str]]:

    article_names = list()
    article_contents = list()

    if not os.path.isfile(jsonPath):
        print('%s is not a file, ommited.' % (jsonPath, ))
        raise ValueError()
    
    print('now opening %s' % (jsonPath, ))

    articles_obj = None
    with open(jsonPath, 'r') as f:
        articles_obj = json.load(f)
    
    # Assuming [ { 'article_name', 'article_content' }]
    for entry in articles_obj:
        article_name = entry['article_name']
        article_content = entry['article_content']

        print_found(article_name, len(article_content))

        article_names.append(article_name)
        article_contents.append(article_content)
    
    print('Found %s articles at total' % (str(len(article_names)), ))
    
    return (
        article_names,
        article_contents,
    )

def read_articles_from_txt(txtPath: str) -> Tuple[List[str], List[str]]:

    if not os.path.isfile(txtPath):
        print('%s is not a text file, omitted.' % txtPath)
        raise ValueError()
    
    print('opening %s' % txtPath)

    fileContent = str
    with open(txtPath, 'r') as f:
        fileContent = f.read()

    allMatches = findall('(.+\n)(.+\n)', fileContent)
    if len(allMatches) == 0:
        print('no articles found.')
        raise ValueError()

    article_names = list()
    article_contents = list()
    for article_name, article_content in allMatches:

        article_name = sub('\n', '', article_name)
        article_content = sub('\n', '', article_content)

        print_found(article_name, len(article_content))
        article_names.append(article_name)
        article_contents.append(article_content)
    
    print('Found %s articles.' % (str(len(article_names)), ))

    return (
        article_names,
        article_contents,
    )

read_articles_from_txt('rmrb.txt')