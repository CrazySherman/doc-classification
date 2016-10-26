import urllib2
import json, os
from datetime import date

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
day = date.today()
print day.year, '--', day.month, '--', day.day
# Notice the shit I use for article delimiters, it should work, absolutely

res = urllib2.urlopen("http://dev.nopical.com/API/breaking-news.json")
jsondata = res.read()
data_today = json.loads(jsondata)
type(data_today)
f = open('ApiArticle.txt','a')
idf = open('ids.txt', 'r+')
id_hash = {}
for line in idf:
    id_hash[int(line)] = True
count = 0
for st in data_today:
    if st["spider_articleText"] and st["id"] not in id_hash:
        f.write('Article Delimiter\\\n')
        f.write(st["spider_articleText"].encode('utf8'))
        f.write('\n')
        idf.write(str(st["id"]) + '\n')
        id_hash[st["id"]] = True
        count += 1
f.close()
idf.close()
print count, ' news articles are loaded'
print float(count) / len(data_today) * 100, "%% scrapes have article\n"
