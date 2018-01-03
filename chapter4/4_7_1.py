import feedparser

ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
ny['entries']
print(len(ny['entries']))
