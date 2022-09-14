from bs4 import BeautifulSoup
import requests as r
import string
import time
from PIL import Image
import random

count = 0
path = r"anime_images\img"

for letter in string.ascii_uppercase:
    for num_pg in range(0, 6000, 50):
        try:
            url = 'https://myanimelist.net/character.php?letter={}&show={}'.format(letter, num_pg)
            page = r.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            char_profiles = soup.find_all("div", {'class': "picSurround"})
            link_to_char = [profile.find("a", href=True) for profile in char_profiles]
            for link in link_to_char:
                photo_pg = r.get(link['href'])
                soup = BeautifulSoup(photo_pg.content, 'html.parser')
                try:
                    image = soup.find('img', {'class': ["portrait-225x350", "lazyloaded"]})
                    img = Image.open(r.get(image['data-src'], stream = True).raw)
                    img.save(str(path) + '_{}.jpg'.format(count))
                    count += 1
                except:
                    pass
                time.sleep(random.randint(0, 3))
        except:
            break