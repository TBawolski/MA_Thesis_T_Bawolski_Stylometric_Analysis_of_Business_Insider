MA Thesis Scripts Appendix

This appendix contains all the scripts used in the MA thesis - Stylometric Analysis of Business Insider Articles Written between 2008 and 2023.
Author: Tomasz Paweł Bawolsku
Album Number: 412303
Supervisor: dr Marcin Opacki

Table of Contents

1. Database Creation Scripts
- 1.1 Database_Creation.py
- 1.2 calculate_word_counts.py
- 1.3 group_articles_by_length.py
- 1.4 scrape_monthly_archives.py
- 1.5 update_length_groups.py
- 1.6 csv_export_script.py

2. Data Cleaning Scripts
- 2.1 check_amp_urls.py
- 2.2 clean_amp_urls.py
- 2.3 clean_summary_regex.py
- 2.4 clean_with_regex.py
- 2.5 extract_author_info.py
- 2.6 extract_first_sentence.py
- 2.7 gramming_analysis_updated.py
- 2.8 jump_to_clean.py
- 2.9 remove_duplicates.py
- 2.10 remove_null_content.py
- 2.11 terms.py

3. Stylometric Analysis Scripts
- 3.1 TAASSC_division.py
- 3.2 TAASSC_results_database.py
- 3.3 lexical_richness.py
- 3.4 lexical_richness_part_2.py
- 3.5 readability_analysis.py

4. Statistical Analysis Scripts (R)
- 4.1 Normality_Tests.R
- 4.2 Regression_Tests_Holm.R
- 4.3 breakout_cohen.R
- 4.4 correlations_with_p_value.R

---

1. Database Creation Scripts

1.1 Database_Creation.py

This production-ready script creates the main database by scraping Business Insider articles with comprehensive error handling and logging.

```python
import sqlite3
import requests
import json
import time
import logging
from lxml import html
from requests.exceptions import RequestException
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
logging.basicConfig(
    filename='database_creation.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
BATCH_SIZE = 1000  
SLEEP_TIME = 1     
MAX_RETRIES = 3    
def setup_session():
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504, 429]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session
def extract_title(tree):
    og_title = tree.xpath('//meta[@property="og:title"]/@content')
    if og_title and og_title[0].strip():
        return og_title[0].strip()
    doc_title = tree.xpath('//title/text()')
    if doc_title and doc_title[0].strip():
        title = doc_title[0].strip()
        if " - Business Insider" in title:
            title = title.replace(" - Business Insider", "")
        return title
    return 'N/A'
def extract_authors(tree):
    authors = []
    meta_authors = tree.xpath('//meta[@name="author"]/@content')
    authors.extend(meta_authors)
    try:
        json_ld = tree.xpath('//script[@type="application/ld+json"]/text()')
        for script in json_ld:
            data = json.loads(script)
            if 'author' in data:
                if isinstance(data['author'], list):
                    authors.extend([a.get('name', '') for a in data['author']])
                elif isinstance(data['author'], dict):
                    authors.append(data['author'].get('name', ''))
    except (json.JSONDecodeError, AttributeError):
        pass
    return authors[0] if authors else 'N/A'
def extract_date(tree):
    pub_time = tree.xpath('//meta[@property="article:published_time"]/@content')
    if pub_time and pub_time[0].strip():
        return pub_time[0].strip()
    return 'N/A'
def extract_content(tree):
    summary_items = tree.xpath('//*[@id="piano-inline-content-wrapper"]/div/div/ul/li/text()')
    summary = "\n".join([item.strip() for item in summary_items if item.strip()])
    content_paragraphs = tree.xpath('//*[@id="piano-inline-content-wrapper"]/div/div/p/text()')
    if not content_paragraphs:
        content_paragraphs = tree.xpath('//div[@class="content-lock-content"]//p/text()')
    content = "\n".join([para.strip() for para in content_paragraphs if para.strip()])
    return summary if summary else 'N/A', content
def scrape_article_data(url, session):
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except RequestException as e:
        logging.error(f'Request failed for {url}: {e}')
        return {'error': f'Request error: {e}'}
    try:
        tree = html.fromstring(response.content)
        title = extract_title(tree)
        author = extract_authors(tree)
        date = extract_date(tree)
        summary, content = extract_content(tree)
        return {
            'title': title,
            'author': author,
            'date': date,
            'summary': summary,
            'content': content,
            'url': url
        }
    except Exception as e:
        logging.error(f'Parsing failed for {url}: {e}')
        return {'error': f'Parsing error: {e}'}
def create_database_and_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            date TEXT,
            summary TEXT,
            content TEXT,
            url TEXT
        )

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON articles(url)')
    conn.commit()
    conn.close()
def main(links_file_path, db_path):
    logging.info('Starting Business Insider database creation')
    create_database_and_table(db_path)
    session = setup_session()
    with open(links_file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    logging.info(f'Processing {len(urls)} URLs')
    total_processed = 0
    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i:i + BATCH_SIZE]
        for url in batch_urls:
            article_data = scrape_article_data(url, session)
            total_processed += 1
            time.sleep(SLEEP_TIME)
if __name__ == '__main__':
    main('article_links.txt', 'business_insider_database.db')
```

1.2 calculate_word_counts.py

```python
import sqlite3
from typing import Tuple
import re
import time
db_path = 'business_insider_database.db' 
table_name = 'articles'  
def add_columns_to_table(cursor):
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN word_count INTEGER;")
    except sqlite3.OperationalError as e:
        print("word_count column already exists:", e)
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN sentence_count INTEGER;")
    except sqlite3.OperationalError as e:
        print("sentence_count column already exists:", e)
def calculate_counts(text: str) -> Tuple[int, int]:
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text)) - 1 
    return word_count, sentence_count
def update_database_with_counts(cursor):
    cursor.execute(f"SELECT id, content FROM {table_name} WHERE word_count IS NULL OR sentence_count IS NULL;")
    articles = cursor.fetchall()
    for article_id, content in articles:
        word_count, sentence_count = calculate_counts(content)
        cursor.execute(f"UPDATE {table_name} SET word_count = ?, sentence_count = ? WHERE id = ?;",
                       (word_count, sentence_count, article_id))
    return len(articles)
def main():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    add_columns_to_table(cursor)
    start_time = time.time()
    processed_count = update_database_with_counts(cursor)
    end_time = time.time()
    conn.commit()
    conn.close()
    print(f"Database update complete. Processed: {processed_count} articles")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
if __name__ == "__main__":
    main()
```

1.3 group_articles_by_length.py

```python
import sqlite3
import os
db_path = 'business_insider_database.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
for year in range(2008, 2024):
    for length_group in range(6):
        directory = os.path.join(str(year), str(length_group))
        if not os.path.exists(directory):
            os.makedirs(directory)
        query = """
        SELECT id, content FROM articles
        WHERE strftime('%Y', date) = ?
        AND length_group = ?

        cursor.execute(query, (str(year), length_group))
        articles = cursor.fetchall()
        for article in articles:
            article_id, content = article
            file_path = os.path.join(directory, f'{article_id}.txt')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
conn.close()
```

1.4 scrape_monthly_archives.py

```python
import requests
from lxml import html
from urllib.parse import urljoin
site_map_url = 'https://www.businessinsider.com/sitemap/html/index.html'
response = requests.get(site_map_url)
tree = html.fromstring(response.content)
links = tree.xpath('//a')
base_url = 'https://www.businessinsider.com/sitemap/html/'
with open('all_links.txt', 'w') as f:
    for link in links:
        href = link.get('href')
        full_url = urljoin(base_url, href)
        f.write(full_url + '\n')
print('All URLs have been saved to all_links.txt')
```

1.5 update_length_groups.py

```python
import sqlite3
database_path = 'business_insider_database.db'
update_query = '''
UPDATE articles
SET length_group = CASE
    WHEN word_count BETWEEN 1 AND 231 THEN 0
    WHEN word_count BETWEEN 232 AND 417 THEN 1
    WHEN word_count BETWEEN 418 AND 633 THEN 2
    WHEN word_count BETWEEN 634 AND 1485 THEN 3 
    WHEN word_count BETWEEN 1486 AND 4531 THEN 4
    WHEN word_count BETWEEN 4534 AND 24428 THEN 5 
END;

try:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(update_query)
    conn.commit()
    print("Length group update was successful.")
except sqlite3.Error as e:
    print("An error occurred:", e)
finally:
    if conn:
        conn.close()
```

---

2. Data Cleaning Scripts

2.1 check_amp_urls.py

```python
import sqlite3
import re
conn = sqlite3.connect('business_insider_database.db')
cursor = conn.cursor()
regex = r"&(?:amp;)+lt;(?:[^&]|&(?:amp;)+[a-z]+;)*?&(?:amp;)+gt;"
cursor.execute("SELECT content FROM articles")
rows = cursor.fetchall()
with open('affected_text.txt', 'w') as file:
    for row in rows:
        content = row[0]
        matches = re.findall(regex, content)
        if matches:
            for match in matches:
                file.write(f"{match}\n")
            file.write("\n")
cursor.close()
conn.close()
```

2.2 clean_amp_urls.py

```python
import sqlite3
import re
conn = sqlite3.connect('business_insider_database.db') 
cursor = conn.cursor()
query = """
SELECT * FROM articles
WHERE content REGEXP '(?i)\\bamp\\W*amp\\W*amp\\b'
LIMIT 10

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None
conn.create_function("REGEXP", 2, regexp)
cursor.execute(query)
results = cursor.fetchall()
for row in results:
    print(row)
cursor.close()
conn.close()
```

2.3 clean_summary_regex.py

```python
import sqlite3
import re
import time
import os
def clean_text(text):
    text = text.strip()
    regexes = [
        r"This is an opinion column\. The thoughts expressed are those of the author\.",
        r"See more stories on Insider's business page\.",
        r"Visit Business Insider's homepage for more stories\.",
        r"Click here for more BI Prime stories\.",
        re.escape("Visit Business Insider's home page for more stories."),
        re.escape("Visit Business Insider's homepage for more stories"),
        re.escape("Visit Business Insider's homepage for more information."),
        re.escape("Click here for more BI Prime articles."),
        re.escape("Click here for more BI Prime content."),
        re.escape("Visit BusinessInsider.com for more stories.")
    ]
    for regex in regexes:
        text = re.sub(regex, '', text)
    read_more_regex = r"Read more.*"
    text = re.sub(read_more_regex, '', text, flags=re.DOTALL)
    return text
def main():
    conn = sqlite3.connect('business_insider_database.db') 
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute('SELECT id, summary FROM articles')  
    articles = cursor.fetchall()
    count = 0
    for article_id, summary in articles:
        cleaned_summary = clean_text(summary)
        cursor.execute('UPDATE articles SET summary = ? WHERE id = ?', (cleaned_summary, article_id))
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} summaries in {time.time() - start_time:.2f} seconds.")
    conn.commit()
    conn.close()
    total_time = time.time() - start_time
    print(f"Total processing time for {count} summaries: {total_time:.2f} seconds.")
    os.system('say "Database cleaning complete."')
if __name__ == "__main__":
    main()
```

2.4 clean_with_regex.py

This comprehensive cleaning script removes various Business Insider-specific text patterns and advertisements from article content.

```python
import sqlite3
import re
import time
import os
def clean_text(text):
    text = text.strip()
    first_sentence = re.match(r'^(.*?\.)\s', text)
    if first_sentence:
        first_sentence = first_sentence.group(1)
        start_second_instance = text.find(first_sentence, len(first_sentence))
        if start_second_instance != -1:  
            text = text[:start_second_instance]
    editorial_note_regex = r"Editorial Note: Any opinions, analyses, reviews or recommendations expressed in this article are those of the author's alone, and have not been reviewed, approved or otherwise endorsed by any card issuer\. Read our editorial standards\..*"
    text = re.sub(editorial_note_regex, '', text, flags=re.DOTALL)
    regexes = [
        r"Advertisement\s+Advertisement",
        re.escape("The Apple Investor is a daily report from SAI. Sign up here to receive it by email."),
        re.escape("When you buy through our links, Insider may earn an affiliate commission. Learn more "),
        re.escape("Visit Business Insider's homepage for more stories."),
        re.escape("Disclosure: Written and researched by the Insider Reviews team."),
    ]
    for regex in regexes:
        text = re.sub(regex, '', text)
    return text
def main():
    conn = sqlite3.connect('business_insider_database.db') 
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute('SELECT id, content FROM articles')  
    articles = cursor.fetchall()
    count = 0
    for article_id, content in articles:
        cleaned_content = clean_text(content)
        cursor.execute('UPDATE articles SET content = ? WHERE id = ?', (cleaned_content, article_id))
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} articles in {time.time() - start_time:.2f} seconds.")
    conn.commit()
    conn.close()
    total_time = time.time() - start_time
    print(f"Total processing time for {count} articles: {total_time:.2f} seconds.")
    os.system('say "Database cleaning complete."')
if __name__ == "__main__":
    main()
```

2.5 extract_author_info.py

```python
import sqlite3
import re
import os
def remove_author_info(text, author_column):
    primary_author = author_column.split(',')[0].strip()
    primary_author = re.escape(primary_author)
    pattern = re.compile(r"\bRead more Read less\b[\s\S]*?$", re.MULTILINE)
    text_before_marker = pattern.split(text)[0]
    author_pattern = rf"({primary_author})[\s\S]*$"
    text_before_author = re.split(author_pattern, text_before_marker, 1)[0]
    return text_before_author.strip()
def process_articles(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, author FROM articles")
    articles = cursor.fetchall()
    print(f"Number of articles fetched: {len(articles)}")
    if not articles:
        print("No articles fetched from the database.")
        return
    changed_article_ids = []
    for article_id, content, author in articles:
        original_content = content
        cleaned_content = remove_author_info(content, author)
        if cleaned_content != original_content:
            cursor.execute("UPDATE articles SET content = ? WHERE id = ?", (cleaned_content, article_id))
            changed_article_ids.append(article_id)
    conn.commit()
    conn.close()
    print(f"Total articles changed: {len(changed_article_ids)}")
db_path = 'business_insider_database.db'
process_articles(db_path)
```

2.6 extract_first_sentence.py

```python
import sqlite3
import re
from collections import defaultdict
def get_first_sentence(text):
    match = re.match(r'^(.*?\.)\s', text)
    if match:
        return match.group(1)
    else:
        return None
def main():
    conn = sqlite3.connect('business_insider_database.db') 
    cursor = conn.cursor()
    cursor.execute('SELECT content FROM articles')  
    articles = cursor.fetchall()
    sentence_count = defaultdict(int)
    for (content,) in articles:
        first_sentence = get_first_sentence(content.strip())
        if first_sentence:
            sentence_count[first_sentence] += 1
    sorted_sentences = sorted(sentence_count.items(), key=lambda x: x[1], reverse=True)
    print("Most frequent first sentences:")
    for sentence, count in sorted_sentences[:100]:  
        print(f"{sentence} - {count} times")
    conn.close()
if __name__ == "__main__":
    main()
```

2.7 gramming_analysis_updated.py

```python
import sqlite3
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
import os
def play_sound():
    os.system('say "Done"')
def tokenizer(text):
    return re.findall(r"\b\w+(?:'\w+)?\b[.,?]*", text)
def process_and_save_ngrams(texts, ngram_range, output_filename):
    vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer, lowercase=False)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sum_words = X.sum(axis=0)
    sum_words_array = sum_words.A.flatten()
    words_freq = [(feature_names[i], sum_words_array[i]) for i in range(len(feature_names))]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words_freq = [item for item in words_freq if item[1] > 4]  
    if words_freq:  
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['N-gram', 'Frequency'])
            writer.writerows(words_freq)
        play_sound()
conn = sqlite3.connect('business_insider_database.db')
cursor = conn.cursor()
cursor.execute("SELECT summary FROM articles")
texts = [row[0] for row in cursor.fetchall() if row[0]]  
for n in range(2, 198):  
    process_and_save_ngrams(texts, (n, n), f'{n}-gram_frequencies.csv')
conn.close()
```

2.8 jump_to_clean.py

```python
import sqlite3
import re
def clean_text(text):
    text = re.sub(r'&(?:amp;)+lt;(?:[^&]|&(?:amp;)+[a-z]+;)*?&(?:amp;)+gt;', '', text)
    if text.startswith("Jump to") and "Read next" not in text:
        text = re.sub(r'^Jump to', '', text)
    pattern = re.compile(r'Jump to(.*?)Read next', re.DOTALL)
    match = pattern.search(text)
    if match:
        content_between = match.group(1)
        if content_between.strip():  
            text = text[:match.start()] + content_between.strip()
        else:
            start_of_following_content = text.find("Read next") + len("Read next")
            text = text[start_of_following_content:]
    return text
conn = sqlite3.connect('business_insider_database.db') 
cursor = conn.cursor()
cursor.execute('SELECT id, content FROM articles')  
articles = cursor.fetchall()
for article_id, content in articles:
    cleaned_content = clean_text(content)
    cursor.execute('UPDATE articles SET content = ? WHERE id = ?', (cleaned_content, article_id))
conn.commit()
conn.close()
print("Database has been cleaned and updated.")
```

2.9 remove_duplicates.py

```python
import sqlite3
def remove_duplicate_articles():
    conn = sqlite3.connect('business_insider_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TEMPORARY TABLE unique_articles AS
            SELECT MIN(id) AS min_id
            FROM articles
            GROUP BY TRIM(content)

    cursor.execute('''
        DELETE FROM articles
        WHERE id NOT IN (SELECT min_id FROM unique_articles)

    conn.commit()
    cursor.execute('SELECT COUNT(*) FROM articles')
    unique_count = cursor.fetchone()[0]
    cursor.execute('DROP TABLE unique_articles')
    conn.close()
    return unique_count
unique_articles_count = remove_duplicate_articles()
print(f"The database now contains {unique_articles_count} unique articles based on content.")
```

2.10 remove_null_content.py

```python
import sqlite3
def remove_empty_content_rows():
    conn = sqlite3.connect('business_insider_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM articles
        WHERE TRIM(content) = '' OR content IS NULL

    conn.commit()
    rows_deleted = cursor.rowcount
    conn.close()
    return rows_deleted
deleted_rows = remove_empty_content_rows()
print(f"Deleted {deleted_rows} rows with empty content.")
```

2.11 terms.py

```python
import sqlite3
def remove_specific_sentence_from_articles():
    conn = sqlite3.connect('business_insider_database.db')
    cursor = conn.cursor()
    sentence_to_remove = "Terms apply to offers listed on this page."
    cursor.execute('''
        UPDATE articles 
        SET content = REPLACE(content, ?, '')
        WHERE content LIKE '%' || ? || '%'

    conn.commit()
    rows_updated = cursor.rowcount
    conn.close()
    return rows_updated
affected_rows = remove_specific_sentence_from_articles()
print(f"Updated {affected_rows} articles to remove the specific sentence.")
```

---

3. Stylometric Analysis Scripts

3.1 TAASSC_division.py

```python
import sqlite3
import os
database_path = 'business_insider_database.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()
years = range(2008, 2024)
for year in years:
    if not os.path.exists(str(year)):
        os.makedirs(str(year))
if not os.path.exists('unknown'):
    os.makedirs('unknown')
query = "SELECT id, content, date, length_group FROM articles"
cursor.execute(query)
articles = cursor.fetchall()
def save_article_to_file(article_id, content, date, length_group):
    try:
        year = date[:4]  
        if year not in map(str, years):
            year = 'unknown'
    except:
        year = 'unknown'
    file_name = f"{article_id}.{length_group}.txt"
    file_path = os.path.join(year, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
for article in articles:
    article_id, content, date, length_group = article
    save_article_to_file(article_id, content, date, length_group)
conn.close()
```

3.2 TAASSC_results_database.py

```python
import pandas as pd
import os
import sqlite3
def detect_delimiter(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ',' in first_line and ';' not in first_line:
            return ','
        elif ';' in first_line and ',' not in first_line:
            return ';'
        else:
            raise ValueError(f"Unexpected delimiter in file: {file_path}")
def split_csv_by_length_group(file_path, year, output_dir, id_column='filename'):
    delimiter = detect_delimiter(file_path)
    df = pd.read_csv(file_path, delimiter=delimiter)
    df['length_group'] = df[id_column].str.split('.').str[1]
    for length_group in df['length_group'].unique():
        length_group_df = df[df['length_group'] == length_group]
        new_file_name = os.path.join(output_dir, f"{year}.{length_group}_results_syntax.csv")
        length_group_df.to_csv(new_file_name, index=False, sep=';')
        print(f"Saved: {new_file_name}")
def create_combined_database(files, db_path, id_column='filename'):
    conn = sqlite3.connect(db_path)
    combined_df = pd.DataFrame()
    for file in files:
        delimiter = detect_delimiter(file)
        year = int(file.split('/')[-1].split('_')[0])
        df = pd.read_csv(file, delimiter=delimiter)
        df['length_group'] = df[id_column].str.split('.').str[1]
        df['year'] = year
        columns = [id_column, 'year', 'length_group'] + list(df.columns[1:-2])
        df = df[columns]
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df.to_sql('results_syntax', conn, if_exists='replace', index=False)
    print("Saved combined_results_syntax to SQLite database")
    conn.close()
input_dir = '/Users/tomaszbawolski/Documents/BusinessInsider/TAASSC_results'
output_dir = os.path.join(input_dir, 'separated_csv_files')
os.makedirs(output_dir, exist_ok=True)
files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) 
         if file.endswith('_results_syntax.csv')]
for file_path in files:
    year = file_path.split('/')[-1].split('_')[0]
    split_csv_by_length_group(file_path, year, output_dir)
db_path = os.path.join(input_dir, 'combined_results_syntax.db')
create_combined_database(files, db_path)
```

3.3 lexical_richness.py

```python
import sqlite3
from lexicalrichness import LexicalRichness
conn = sqlite3.connect('/Users/tomaszbawolski/Documents/BusinessInsider/business_insider_databse.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS lexical_richness_scores (
    id INTEGER PRIMARY KEY,
    title TEXT,
    date TEXT,
    url TEXT,
    ttr REAL,
    rttr REAL,
    cttr REAL,
    herdan REAL,
    dugast REAL,
    maas REAL,
    FOREIGN KEY(id) REFERENCES articles(id)
);

cursor.execute("SELECT id, title, date, content, url FROM articles")
articles = cursor.fetchall()
insert_query = '''INSERT INTO lexical_richness_scores (id, title, date, url, ttr, rttr, cttr, herdan, dugast, maas) 
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
processed_count = 0 
for article in articles:
    id, title, date, content, url = article
    if not content.strip():
        print(f"Skipped article {id} due to empty content.")
        continue
    lex = LexicalRichness(content)
    ttr, rttr, cttr, herdan, dugast, maas = (0, 0, 0, 0, 0, 0)
    try:
        ttr = lex.ttr  
        rttr = lex.rttr  
        cttr = lex.cttr  
        herdan = lex.Herdan  
        dugast = lex.Dugast  
        maas = lex.Maas  
    except ZeroDivisionError:
        print(f"Error calculating scores for article {id}; using default values.")
    cursor.execute(insert_query, (id, title, date, url, ttr, rttr, cttr, herdan, dugast, maas))
    processed_count += 1  
    if processed_count % 25000 == 0: 
        print(f"Processed {processed_count} articles.")
conn.commit()
conn.close()
```

3.4 lexical_richness_part_2.py

```python
import sqlite3
import math
import logging
from lexicalrichness import LexicalRichness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
conn = sqlite3.connect('/Users/tomaszbawolski/Documents/BusinessInsider/business_insider_databse.db')
cursor = conn.cursor()
cursor.execute("SELECT id, content FROM articles WHERE length_group != 0")
articles = cursor.fetchall()
update_query = '''UPDATE lexical_richness_scores
                  SET guiraud = ?, msttr = ?, mattr = ?, mtld = ?, hdd = ?, vocd = ?, yules_k = ?
                  WHERE id = ?'''
batch_size = 10000
total_articles = len(articles)
num_batches = (total_articles + batch_size - 1) // batch_size
for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min(start_index + batch_size, total_articles)
    batch_articles = articles[start_index:end_index]
    logging.info(f"Processing batch {batch_num+1}/{num_batches} ({start_index+1} to {end_index})")
    for article in batch_articles:
        article_id, content = article
        if not content.strip():
            continue
        lex = LexicalRichness(content)
        try:
            guiraud = lex.terms / math.sqrt(lex.words) if lex.words > 0 else 0
            msttr = lex.msttr(segment_window=50)
            mattr = lex.mattr(window_size=50)
            mtld = lex.mtld(threshold=0.72)
            hdd = lex.hdd(draws=42)
            vocd = lex.vocd(ntokens=50, within_sample=100, iterations=3)
            yules_k = lex.yulek
        except Exception as e:
            guiraud, msttr, mattr, mtld, hdd, vocd, yules_k = (0, 0, 0, 0, 0, 0, 0)
            logging.error(f"Error calculating metrics for article {article_id}: {e}")
        cursor.execute(update_query, (guiraud, msttr, mattr, mtld, hdd, vocd, yules_k, article_id))
    conn.commit() 
    logging.info(f"Finished processing batch {batch_num+1}/{num_batches}")
conn.close()
```

3.5 readability_analysis.py

This script calculates comprehensive readability metrics for all articles using the textstat library.

```python
import sqlite3
import logging
import textstat
import nltk
import ssl
try:
    nltk.download('cmudict', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def calculate_readability_metrics(text):

    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)
        coleman_liau_index = textstat.coleman_liau_index(text)
        automated_readability_index = textstat.automated_readability_index(text)
        dale_chall_readability_score = textstat.dale_chall_readability_score(text)
        mcalpine_eflaw = textstat.mcalpine_eflaw(text)
        return (flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index,
                coleman_liau_index, automated_readability_index, dale_chall_readability_score,
                mcalpine_eflaw)
    except Exception as e:
        logging.error(f"Error calculating readability metrics: {e}")
        return (0, 0, 0, 0, 0, 0, 0, 0)
def main():
    conn = sqlite3.connect('business_insider_databse.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, date, url, content, length_group FROM articles WHERE length_group != 0")
    articles = cursor.fetchall()
    cursor.execute("DROP TABLE IF EXISTS readability_scores")
    cursor.execute('''CREATE TABLE readability_scores (
        id INTEGER PRIMARY KEY,
        title TEXT,
        date TEXT,
        url TEXT,
        flesch_reading_ease REAL,
        flesch_kincaid_grade REAL,
        gunning_fog REAL,
        smog_index REAL,
        coleman_liau_index REAL,
        automated_readability_index REAL,
        dale_chall_readability_score REAL,
        mcalpine_eflaw REAL,
        length_group INTEGER,
        FOREIGN KEY(id) REFERENCES articles(id)
    )''')
    insert_query = '''INSERT INTO readability_scores 
                      (id, title, date, url, flesch_reading_ease, flesch_kincaid_grade, 
                       gunning_fog, smog_index, coleman_liau_index, automated_readability_index,
                       dale_chall_readability_score, mcalpine_eflaw, length_group)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    batch_size = 10000
    total_articles = len(articles)
    num_batches = (total_articles + batch_size - 1) // batch_size
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, total_articles)
        batch_articles = articles[start_index:end_index]
        logging.info(f"Processing batch {batch_num+1}/{num_batches} ({start_index+1} to {end_index})")
        for article in batch_articles:
            article_id, title, date, url, content, length_group = article
            if not content or not content.strip():
                logging.warning(f"Skipping article {article_id} - empty content")
                continue
            metrics = calculate_readability_metrics(content)
            cursor.execute(insert_query, (article_id, title, date, url, *metrics, length_group))
        conn.commit()
        logging.info(f"Finished processing batch {batch_num+1}/{num_batches}")
    conn.close()
    logging.info("Readability analysis completed successfully")
if __name__ == "__main__":
    main()
```

---

4. Statistical Analysis Scripts (R)

4.1 Normality_Tests.R

Comprehensive model diagnostics including normality testing, homoscedasticity checks, and residual analysis for all regression models.

```r
rm(list = ls())
options(scipen = 999, digits = 4)
suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(RSQLite)
  library(ggplot2)
  library(gridExtra)
  library(lmtest)
  library(car)
})
root_dir <- getwd()
data_dir <- file.path(root_dir, "Data_Bases")
out_dir  <- file.path(root_dir, "03_Analysis_Outputs", "MODEL_DIAGNOSTICS")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
cat("=== MODEL DIAGNOSTICS ANALYSIS ===\n")
con_main   <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "business_insider_database.db"))
con_syntax <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "combined_results_syntax.db"))
lexical_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group, vocd, maas, ttr
FROM lexical_richness_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
readability_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group,
       flesch_reading_ease, coleman_liau_index, smog_index
FROM readability_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
syntax_data <- setDT(dbGetQuery(con_syntax, "
SELECT year, length_group, advmod_per_cl, dep_per_cl, acomp_per_cl
FROM results_syntax
WHERE CAST(year AS INTEGER) BETWEEN 2008 AND 2023
  AND CAST(length_group AS INTEGER) IN (1,2,3)
"))
dbDisconnect(con_main); dbDisconnect(con_syntax)
syntax_data[, year := as.integer(year)]
syntax_data[, length_group := as.integer(length_group)]
perform_diagnostics <- function(data, variable, group, family_name) {
  yearly_data <- data[length_group == group & !is.na(get(variable)), .(
    mean_value = mean(get(variable), na.rm = TRUE),
    n_obs = .N
  ), by = year][order(year)]
  yearly_data <- yearly_data[n_obs >= 10]
  if(nrow(yearly_data) < 5) return(NULL)
  model <- lm(mean_value ~ year, data = yearly_data)
  residuals <- residuals(model)
  fitted_vals <- fitted(model)
  n_obs <- nrow(yearly_data)
  shapiro_test <- if(n_obs <= 50) shapiro.test(residuals) else NULL
  ks_test <- ks.test(residuals, "pnorm", mean=mean(residuals), sd=sd(residuals))
  bp_test <- bptest(model)
  cooksd <- cooks.distance(model)
  leverage <- hatvalues(model)
  cook_threshold <- 4/(n_obs - 2)
  high_cook <- sum(cooksd > cook_threshold)
  leverage_threshold <- 2*2/n_obs
  high_leverage <- sum(leverage > leverage_threshold)
  model_summary <- summary(model)
  r_squared <- model_summary$r.squared
  adj_r_squared <- model_summary$adj.r.squared
  diagnostics <- data.table(
    family = family_name,
    variable = variable,
    length_group = group,
    n_years = n_obs,
    shapiro_w = if(!is.null(shapiro_test)) shapiro_test$statistic else NA,
    shapiro_p = if(!is.null(shapiro_test)) shapiro_test$p.value else NA,
    ks_d = ks_test$statistic,
    ks_p = ks_test$p.value,
    bp_statistic = bp_test$statistic,
    bp_p = bp_test$p.value,
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    high_cook_n = high_cook,
    high_leverage_n = high_leverage,
    max_cooksd = max(cooksd),
    max_leverage = max(leverage),
    residual_mean = mean(residuals),
    residual_sd = sd(residuals)
  )
  return(list(
    diagnostics = diagnostics,
    model = model,
    yearly_data = yearly_data
  ))
}
datasets <- list(
  list(data = lexical_data, vars = c("vocd","maas","ttr"), family = "Lexical richness"),
  list(data = readability_data, vars = c("flesch_reading_ease","coleman_liau_index","smog_index"), family = "Readability"),
  list(data = syntax_data, vars = c("advmod_per_cl", "dep_per_cl", "acomp_per_cl"), family = "Syntactic complexity")
)
all_diagnostics <- list()
models_for_plots <- list()
for(ds in datasets) {
  family_name <- ds$family
  cat("Analyzing", family_name, "variables...\n")
  for(var in ds$vars) {
    for(lg in 1:3) {
      result <- perform_diagnostics(ds$data, var, lg, family_name)
      if(!is.null(result)) {
        key <- paste(family_name, var, lg, sep = "::")
        all_diagnostics[[key]] <- result$diagnostics
        models_for_plots[[key]] <- list(
          model = result$model,
          yearly_data = result$yearly_data,
          variable = var,
          length_group = lg,
          family = family_name
        )
      }
    }
  }
}
diagnostics_results <- rbindlist(all_diagnostics, use.names = TRUE, fill = TRUE)
create_diagnostic_plots <- function(model_info, save_path) {
  model <- model_info$model
  var_name <- model_info$variable
  lg <- model_info$length_group
  family <- model_info$family
  p1 <- ggplot(data.frame(fitted = fitted(model), residuals = residuals(model)), 
               aes(x = fitted, y = residuals)) +
    geom_point() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_smooth(se = FALSE, color = "red") +
    labs(title = "Residuals vs Fitted", x = "Fitted values", y = "Residuals") +
    theme_minimal()
  p2 <- ggplot(data.frame(sample = residuals(model)), aes(sample = sample)) +
    stat_qq() + stat_qq_line() +
    labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal()
  p3 <- ggplot(data.frame(fitted = fitted(model), 
                          sqrt_resid = sqrt(abs(residuals(model)))), 
               aes(x = fitted, y = sqrt_resid)) +
    geom_point() +
    geom_smooth(se = FALSE, color = "red") +
    labs(title = "Scale-Location", x = "Fitted values", y = "√|Residuals|") +
    theme_minimal()
  p4 <- ggplot(data.frame(obs = 1:length(cooks.distance(model)), 
                          cooksd = cooks.distance(model)), 
               aes(x = obs, y = cooksd)) +
    geom_col() +
    geom_hline(yintercept = 4/(nrow(model_info$yearly_data)-2), linetype = "dashed", color = "red") +
    labs(title = "Cook's Distance", x = "Observation", y = "Cook's Distance") +
    theme_minimal()
  combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2,
                                top = paste(family, "-", var_name, "- Length Group", lg))
  ggsave(save_path, combined_plot, width = 12, height = 10, dpi = 300)
  return(combined_plot)
}
key_vars <- c("vocd", "flesch_reading_ease", "advmod_per_cl")
for(var in key_vars) {
  for(lg in 1:3) {
    model_key <- NULL
    for(key in names(models_for_plots)) {
      if(grepl(var, key) && grepl(paste0("::", lg, "$"), key)) {
        model_key <- key
        break
      }
    }
    if(!is.null(model_key)) {
      model_info <- models_for_plots[[model_key]]
      save_path <- file.path(out_dir, paste0("diagnostics_", var, "_group", lg, ".png"))
      create_diagnostic_plots(model_info, save_path)
    }
  }
}
diagnostics_results[, `:=`(
  normality_ok = (is.na(shapiro_p) | shapiro_p > 0.05) & (ks_p > 0.05),
  homoscedasticity_ok = bp_p > 0.05,
  good_fit = r_squared > 0.3,
  outliers_present = high_cook_n > 0 | high_leverage_n > 0
)]
family_summary <- diagnostics_results[, .(
  n_models = .N,
  normality_violations = sum(!normality_ok, na.rm = TRUE),
  homoscedasticity_violations = sum(!homoscedasticity_ok, na.rm = TRUE),
  good_fits = sum(good_fit, na.rm = TRUE),
  models_with_outliers = sum(outliers_present, na.rm = TRUE),
  mean_r_squared = mean(r_squared, na.rm = TRUE)
), by = family]
fwrite(diagnostics_results, file.path(out_dir, "detailed_diagnostics.csv"))
fwrite(family_summary, file.path(out_dir, "diagnostics_summary_by_family.csv"))
cat("\n=== MODEL DIAGNOSTICS COMPLETE ===\n")
```

4.2 Regression_Tests_Holm.R

Focused regression analysis with Holm-Bonferroni correction for multiple testing.

```r
rm(list = ls())
options(scipen = 999, digits = 4)
suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(RSQLite)
  library(writexl)
})
root_dir <- getwd()
data_dir <- file.path(root_dir, "Data_Bases")
out_dir  <- file.path(root_dir, "03_Analysis_Outputs", "HOLM_FOCUSED_ANALYSIS")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
YEARS_SPAN <- 2023 - 2008
con_main   <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "business_insider_database.db"))
con_syntax <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "combined_results_syntax.db"))
lexical_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group, vocd, maas, ttr
FROM lexical_richness_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
readability_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group,
       flesch_reading_ease, coleman_liau_index, smog_index
FROM readability_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
syntax_data <- setDT(dbGetQuery(con_syntax, "
SELECT *
FROM results_syntax
WHERE CAST(year AS INTEGER) BETWEEN 2008 AND 2023
  AND CAST(length_group AS INTEGER) IN (1,2,3)
"))
dbDisconnect(con_main); dbDisconnect(con_syntax)
syntax_data[, year := as.integer(year)]
syntax_data[, length_group := as.integer(length_group)]
analyze_trend <- function(data, variable, group) {
  yearly_means <- data[length_group == group & !is.na(get(variable)), .(
    mean_value = mean(get(variable), na.rm = TRUE),
    n_obs = .N
  ), by = year][order(year)]
  yearly_means <- yearly_means[n_obs >= 10]
  if (nrow(yearly_means) < 5) return(NULL)
  model <- lm(mean_value ~ year, data = yearly_means)
  summary_model <- summary(model)
  slope <- unname(coef(model)["year"])
  r2 <- summary_model$r.squared
  p_val <- summary_model$coefficients["year", "Pr(>|t|)"]
  data.table(
    variable = variable,
    length_group = group,
    trend_slope = as.numeric(slope),
    trend_r_squared = as.numeric(r2),
    trend_p_value = as.numeric(p_val),
    delta_2008_2023 = as.numeric(slope * YEARS_SPAN)
  )
}
readability_vars <- c("flesch_reading_ease", "coleman_liau_index", "smog_index")
lexical_vars <- c("vocd", "maas", "ttr")
syntax_vars <- c("acomp_per_cl", "advmod_per_cl", "dep_per_cl")
datasets <- list(
  list(data = lexical_data, vars = lexical_vars, family = "Lexical Richness"),
  list(data = readability_data, vars = readability_vars, family = "Readability"),
  list(data = syntax_data, vars = syntax_vars, family = "Syntactic Complexity")
)
results_list <- list()
for (dataset in datasets) {
  family_name <- dataset$family
  for (variable in dataset$vars) {
    for (length_group in 1:3) {
      result <- analyze_trend(dataset$data, variable, length_group)
      if (!is.null(result)) {
        result[, family := family_name]
        results_list[[paste(family_name, variable, length_group, sep = "::")]] <- result
      }
    }
  }
}
all_trends <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
setorder(all_trends, family, variable, length_group)
all_trends[, p_adj_holm := p.adjust(trend_p_value, method = "holm")]
pretty_names <- function(x) {
  name_map <- c(
    flesch_reading_ease = "Flesch Reading Ease",
    coleman_liau_index = "Coleman–Liau Index",
    smog_index = "SMOG Index",
    vocd = "VOCD", maas = "Maas", ttr = "TTR",
    acomp_per_cl = "Adjective Complements per Clause",
    advmod_per_cl = "Adverbial Modifiers per Clause",
    dep_per_cl = "Undefined Dependents per Clause"
  )
  ifelse(x %in% names(name_map), name_map[x], x)
}
group_names <- c(`1` = "Short", `2` = "Medium", `3` = "Long")
significance_flags <- function(p) {
  ifelse(is.na(p), "ns",
         ifelse(p < 0.001, "***",
                ifelse(p < 0.01, "**",
                       ifelse(p < 0.05, "*", "ns"))))
}
final_results <- all_trends[, .(
  Family = family,
  Variable = pretty_names(variable),
  Group = group_names[as.character(length_group)],
  `Trend Direction` = ifelse(trend_slope > 0, "▲", "▼"),
  `Δ 2008→2023` = sprintf("%.3f", delta_2008_2023),
  `R²` = sprintf("%.3f", trend_r_squared),
  `p-value (raw)` = formatC(trend_p_value, format = "g", digits = 3),
  `p-value (Holm Adj.)` = formatC(p_adj_holm, format = "g", digits = 3),
  `Significance` = significance_flags(p_adj_holm)
)][order(Family, Variable, Group)]
results_csv <- file.path(out_dir, "focused_analysis_results.csv")
results_xlsx <- file.path(out_dir, "focused_analysis_results.xlsx")
fwrite(final_results, results_csv)
write_xlsx(list("Holm_Focused_Results" = as.data.frame(final_results)), results_xlsx)
total_tests <- nrow(all_trends)
sig_raw <- sum(all_trends$trend_p_value < 0.05, na.rm = TRUE)
sig_holm <- sum(all_trends$p_adj_holm < 0.05, na.rm = TRUE)
cat(sprintf("Total tests: %d\n", total_tests))
cat(sprintf("Significant (raw): %d\n", sig_raw))
cat(sprintf("Significant (Holm-adjusted): %d\n", sig_holm))
```

4.3 breakout_cohen.R

Breakpoint detection analysis using t-tests with corrected Cohen's d effect sizes.

```r
rm(list = ls())
options(scipen = 999, digits = 4)
suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(RSQLite)
  library(ggplot2)
})
root_dir <- getwd()
data_dir <- file.path(root_dir, "Data_Bases") 
out_dir <- file.path(root_dir, "03_Analysis_Outputs", "BREAKPOINT_ANALYSIS")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
con_main <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "business_insider_database.db"))
con_syntax <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "combined_results_syntax.db"))
lexical_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group, vocd, maas, ttr
FROM lexical_richness_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
readability_data <- setDT(dbGetQuery(con_main, "
SELECT CAST(substr(date,1,4) AS INTEGER) AS year, length_group,
       flesch_reading_ease, coleman_liau_index, smog_index
FROM readability_scores
WHERE length_group IN (1,2,3) AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
syntax_data <- setDT(dbGetQuery(con_syntax, "
SELECT year, length_group, advmod_per_cl, dep_per_cl, acomp_per_cl
FROM results_syntax
WHERE CAST(year AS INTEGER) BETWEEN 2008 AND 2023
  AND CAST(length_group AS INTEGER) IN (1,2,3)
"))
dbDisconnect(con_main); dbDisconnect(con_syntax)
syntax_data[, year := as.integer(year)]
syntax_data[, length_group := as.integer(length_group)]
calculate_cohens_d_from_raw_data <- function(data, variable, group, break_year) {
  pre_data <- data[length_group == group & year <= break_year, get(variable)]
  post_data <- data[length_group == group & year > break_year, get(variable)]
  pre_data <- pre_data[!is.na(pre_data)]
  post_data <- post_data[!is.na(post_data)]
  if (length(pre_data) < 2 || length(post_data) < 2) {
    return(NA)
  }
  mean_pre <- mean(pre_data)
  mean_post <- mean(post_data)
  n_pre <- length(pre_data)
  n_post <- length(post_data)
  sd_pre <- sd(pre_data)
  sd_post <- sd(post_data)
  pooled_sd <- sqrt(((n_pre - 1) * sd_pre^2 + (n_post - 1) * sd_post^2) / (n_pre + n_post - 2))
  cohens_d <- if (!is.na(pooled_sd) && pooled_sd > 0) {
    (mean_post - mean_pre) / pooled_sd
  } else {
    NA
  }
  return(cohens_d)
}
detect_breakpoint_ttest <- function(data, variable, group, min_years = 3) {
  yearly_data <- data[length_group == group & !is.na(get(variable)), .(
    mean_value = mean(get(variable), na.rm = TRUE),
    n_obs = .N
  ), by = year][order(year)]
  yearly_data <- yearly_data[n_obs >= 10]
  if(nrow(yearly_data) < 6) return(NULL)
  years <- yearly_data$year
  values <- yearly_data$mean_value
  potential_breaks <- years[3:(length(years)-2)]
  results <- list()
  for(break_year in potential_breaks) {
    pre_idx <- which(years <= break_year)
    post_idx <- which(years > break_year)
    if(length(pre_idx) >= min_years && length(post_idx) >= min_years) {
      pre_values <- values[pre_idx]
      post_values <- values[post_idx]
      if(length(pre_values) > 1 && length(post_values) > 1) {
        t_test <- t.test(post_values, pre_values)
        results[[as.character(break_year)]] <- data.table(
          variable = variable,
          length_group = group,
          breakpoint_year = break_year,
          pre_mean_of_averages = mean(pre_values),
          post_mean_of_averages = mean(post_values),
          mean_diff = mean(post_values) - mean(pre_values),
          t_statistic = t_test$statistic,
          p_value = t_test$p.value
        )
      }
    }
  }
  if(length(results) == 0) return(NULL)
  breakpoint_results <- rbindlist(results)
  best_break <- breakpoint_results[which.min(p_value)]
  correct_cohens_d <- calculate_cohens_d_from_raw_data(data, variable, group, best_break$breakpoint_year)
  best_break[, cohens_d := correct_cohens_d]
  return(best_break)
}
test_vars <- list(
  syntax = list(data = syntax_data, vars = c("advmod_per_cl", "dep_per_cl", "acomp_per_cl")),
  lexical = list(data = lexical_data, vars = c("vocd", "maas", "ttr")),
  readability = list(data = readability_data, vars = c("flesch_reading_ease", "coleman_liau_index", "smog_index"))
)
all_ttest_results <- list()
for(family_name in names(test_vars)) {
  family_data <- test_vars[[family_name]]
  cat("Analyzing", family_name, "variables...\n")
  for(var in family_data$vars) {
    for(lg in 1:3) {
      ttest_result <- detect_breakpoint_ttest(family_data$data, var, lg)
      if(!is.null(ttest_result)) {
        ttest_result[, family := family_name]
        all_ttest_results[[paste(family_name, var, lg, sep = "::")]] <- ttest_result
      }
    }
  }
}
ttest_breaks <- rbindlist(all_ttest_results, use.names = TRUE, fill = TRUE)
if(nrow(ttest_breaks) > 0) {
  ttest_breaks[, p_adj := p.adjust(p_value, method = "holm")]
  ttest_breaks[, significant := p_adj < 0.05]
  ttest_breaks_formatted <- copy(ttest_breaks)
  cols_to_round <- c("pre_mean_of_averages", "post_mean_of_averages", "mean_diff", "cohens_d", "t_statistic")
  ttest_breaks_formatted[, (cols_to_round) := lapply(.SD, round, 3), .SDcols = cols_to_round]
  final_output <- ttest_breaks_formatted[, .(
    family, variable, length_group, breakpoint_year, 
    mean_diff, cohens_d, t_statistic, p_value, p_adj, significant
  )]
  fwrite(final_output, file.path(out_dir, "ttest_breakpoints_corrected.csv"))
  significant_breaks <- ttest_breaks[significant == TRUE]
  if(nrow(significant_breaks) > 0) {
    cat("\nSignificant breakpoints detected:\n")
    print(significant_breaks[, .(variable, length_group, breakpoint_year, mean_diff, cohens_d)])
  }
}
cat("\n=== BREAKPOINT ANALYSIS COMPLETE ===\n")
```

4.4 correlations_with_p_value.R

Comprehensive correlation analysis with significance testing and multiple comparison corrections.

```r
rm(list = ls())
options(scipen = 999, digits = 4)
suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(RSQLite)
  library(corrplot)
  library(ggplot2)
  library(writexl)
})
root_dir <- getwd()
data_dir <- file.path(root_dir, "Data_Bases")
output_dir <- file.path(root_dir, "03_Analysis_Outputs", "CORRELATION_ANALYSIS_WITH_SIGNIFICANCE")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
cat("CORRELATION ANALYSIS WITH SIGNIFICANCE\n")
con_main <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "business_insider_database.db"))
con_syntax <- dbConnect(RSQLite::SQLite(), file.path(data_dir, "combined_results_syntax.db"))
lexical_data <- setDT(dbGetQuery(con_main, "
SELECT 
  CAST(substr(date,1,4) AS INTEGER) AS year, 
  length_group, 
  vocd, 
  maas, 
  ttr
FROM lexical_richness_scores
WHERE length_group IN (1,2,3) 
  AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
readability_data <- setDT(dbGetQuery(con_main, "
SELECT 
  CAST(substr(date,1,4) AS INTEGER) AS year, 
  length_group,
  flesch_reading_ease, 
  coleman_liau_index, 
  smog_index
FROM readability_scores
WHERE length_group IN (1,2,3) 
  AND CAST(substr(date,1,4) AS INTEGER) BETWEEN 2008 AND 2023
"))
syntax_data <- setDT(dbGetQuery(con_syntax, "
SELECT 
  CAST(year AS INTEGER) as year,
  CAST(length_group AS INTEGER) as length_group,
  acomp_per_cl,
  advmod_per_cl,
  dep_per_cl
FROM results_syntax
WHERE CAST(year AS INTEGER) BETWEEN 2008 AND 2023
  AND CAST(length_group AS INTEGER) IN (1,2,3)
"))
dbDisconnect(con_main)
dbDisconnect(con_syntax)
lexical_yearly <- lexical_data[, .(
  vocd = mean(vocd, na.rm = TRUE),
  maas = mean(maas, na.rm = TRUE),
  ttr = mean(ttr, na.rm = TRUE)
), by = .(year, length_group)]
readability_yearly <- readability_data[, .(
  flesch_reading_ease = mean(flesch_reading_ease, na.rm = TRUE),
  coleman_liau_index = mean(coleman_liau_index, na.rm = TRUE),
  smog_index = mean(smog_index, na.rm = TRUE)
), by = .(year, length_group)]
syntax_yearly <- syntax_data[, .(
  acomp_per_cl = mean(acomp_per_cl, na.rm = TRUE),
  advmod_per_cl = mean(advmod_per_cl, na.rm = TRUE),
  dep_per_cl = mean(dep_per_cl, na.rm = TRUE)
), by = .(year, length_group)]
correlation_data <- lexical_yearly[readability_yearly, on = .(year, length_group)]
correlation_data <- correlation_data[syntax_yearly, on = .(year, length_group)]
vars_to_analyze <- c("vocd", "maas", "ttr", 
                     "flesch_reading_ease", "coleman_liau_index", "smog_index",
                     "acomp_per_cl", "advmod_per_cl", "dep_per_cl")
var_names <- c(
  vocd = "VOCD",
  maas = "Maas Index",
  ttr = "TTR",
  flesch_reading_ease = "Flesch Reading Ease",
  coleman_liau_index = "Coleman-Liau Index",
  smog_index = "SMOG Index",
  acomp_per_cl = "Adj. Complements/Clause",
  advmod_per_cl = "Adv. Modifiers/Clause",
  dep_per_cl = "Undef. Dependents/Clause"
)
calculate_cor_p_values <- function(data, vars) {
  n <- nrow(data)
  cor_mat <- cor(data[, ..vars], use = "complete.obs")
  p_mat <- matrix(NA, nrow = length(vars), ncol = length(vars))
  rownames(p_mat) <- vars
  colnames(p_mat) <- vars
  for(i in 1:length(vars)) {
    for(j in 1:length(vars)) {
      if(i != j) {
        r <- cor_mat[i, j]
        t_stat <- r * sqrt((n - 2) / (1 - r^2))
        p_mat[i, j] <- 2 * pt(abs(t_stat), df = n - 2, lower.tail = FALSE)
      } else {
        p_mat[i, j] <- NA
      }
    }
  }
  return(list(cor = cor_mat, p_values = p_mat, n = n))
}
correlation_results <- list()
p_value_results <- list()
correlation_tests_all <- list()
group_names <- c("1" = "Short Articles (232-417 words)",
                 "2" = "Medium Articles (418-633 words)",
                 "3" = "Long Articles (634-1,485 words)")
for(lg in 1:3) {
  cat(sprintf("Analyzing Group %d (%s)...\n", lg, group_names[as.character(lg)]))
  group_data <- correlation_data[length_group == lg]
  results <- calculate_cor_p_values(group_data, vars_to_analyze)
  cor_matrix <- results$cor
  p_matrix <- results$p_values
  rownames(cor_matrix) <- var_names[rownames(cor_matrix)]
  colnames(cor_matrix) <- var_names[colnames(cor_matrix)]
  rownames(p_matrix) <- var_names[rownames(p_matrix)]
  colnames(p_matrix) <- var_names[colnames(p_matrix)]
  correlation_results[[paste0("group_", lg)]] <- cor_matrix
  p_value_results[[paste0("group_", lg)]] <- p_matrix
  upper_tri_indices <- which(upper.tri(p_matrix), arr.ind = TRUE)
  for(k in 1:nrow(upper_tri_indices)) {
    i <- upper_tri_indices[k, 1]
    j <- upper_tri_indices[k, 2]
    correlation_tests_all[[length(correlation_tests_all) + 1]] <- data.frame(
      Group = lg,
      Var1 = rownames(p_matrix)[i],
      Var2 = colnames(p_matrix)[j],
      Correlation = cor_matrix[i, j],
      P_Value_Raw = p_matrix[i, j],
      stringsAsFactors = FALSE
    )
  }
  png(file.path(output_dir, sprintf("correlation_matrix_group_%d.png", lg)), 
      width = 1200, height = 1200, res = 150)
  corrplot(cor_matrix,
           method = "color",
           type = "full",
           tl.col = "black",
           tl.srt = 45,
           tl.cex = 0.8,
           addCoef.col = "black",
           number.cex = 0.6,
           number.digits = 3,
           col = colorRampPalette(c("#4575B4", "#D1E5F0", "white", "#FDDBC7", "#D73027"))(100),
           title = sprintf("Correlation Matrix - %s", group_names[as.character(lg)]),
           mar = c(0, 0, 2, 0))
  dev.off()
}
all_tests_df <- do.call(rbind, correlation_tests_all)
all_tests_df$P_Value_Holm <- p.adjust(all_tests_df$P_Value_Raw, method = "holm")
all_tests_df$Sig_Raw <- ifelse(all_tests_df$P_Value_Raw < 0.001, "***",
                               ifelse(all_tests_df$P_Value_Raw < 0.01, "**",
                                      ifelse(all_tests_df$P_Value_Raw < 0.05, "*", "ns")))
all_tests_df$Sig_Holm <- ifelse(all_tests_df$P_Value_Holm < 0.001, "***",
                                ifelse(all_tests_df$P_Value_Holm < 0.01, "**",
                                       ifelse(all_tests_df$P_Value_Holm < 0.05, "*", "ns")))
write.csv(all_tests_df,
          file.path(output_dir, "all_correlation_tests_with_holm_correction.csv"),
          row.names = FALSE)
overall_results <- calculate_cor_p_values(correlation_data, vars_to_analyze)
overall_cor <- overall_results$cor
overall_p <- overall_results$p_values
rownames(overall_cor) <- var_names[rownames(overall_cor)]
colnames(overall_cor) <- var_names[colnames(overall_cor)]
png(file.path(output_dir, "correlation_matrix_overall.png"), 
    width = 1400, height = 1400, res = 150)
corrplot(overall_cor,
         method = "color",
         type = "full",
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.9,
         addCoef.col = "black",
         number.cex = 0.7,
         number.digits = 3,
         col = colorRampPalette(c("#4575B4", "#D1E5F0", "white", "#FDDBC7", "#D73027"))(100),
         title = "Overall Correlation Matrix (All Article Groups)",
         mar = c(0, 0, 2, 0))
dev.off()
strong_correlations <- all_tests_df[abs(all_tests_df$Correlation) > 0.7, ]
strong_correlations <- strong_correlations[order(abs(strong_correlations$Correlation), decreasing = TRUE), ]
if(nrow(strong_correlations) > 0) {
  write.csv(strong_correlations,
            file.path(output_dir, "strong_correlations_with_significance.csv"),
            row.names = FALSE)
}
n_total_tests <- nrow(all_tests_df)
n_sig_raw <- sum(all_tests_df$P_Value_Raw < 0.05)
n_sig_holm <- sum(all_tests_df$P_Value_Holm < 0.05)
n_strong <- sum(abs(all_tests_df$Correlation) > 0.7)
n_strong_sig_holm <- sum(abs(all_tests_df$Correlation) > 0.7 & all_tests_df$P_Value_Holm < 0.05)
cat(sprintf("\nSummary: %d/%d correlations significant after Holm correction\n", 
            n_sig_holm, n_total_tests))
cat(sprintf("%d/%d strong correlations remain significant\n", 
            n_strong_sig_holm, n_strong))
```