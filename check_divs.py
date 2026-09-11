with open('frontend/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('id="tab-overview"')
end = text.find('id="tab-anomaly-analytics"')
overview_html = text[start:end]

div_count = overview_html.count('<div') - overview_html.count('</div')
print('Unmatched div count:', div_count)
