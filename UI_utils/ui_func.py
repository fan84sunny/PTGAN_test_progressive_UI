
import cv2
from PIL import Image, ImageQt


def get_one_img(query_img_path, transform):

    img = Image.open(query_img_path).convert("RGB")
    img = transform(img)
    pid = int(query_img_path[-24:-20])
    camid = int(query_img_path[-18:-15])

    return {'origin': img,
            'pid': pid,
            'camid': camid,
            'trackid': -1,
            'file_name': query_img_path
            }


def ui_result(cur_result, k = 100):
    topK_result = cur_result[:k]
    bg = Image.new('RGBA', (150 * 5, 150 * (k // 5)), '#FFFFFF')
    for i, result in enumerate(topK_result):
        img_path = result[0]
        img_dist = result[1]
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (150, 150))
        cv2.putText(im, img_path.split('/')[-1], (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 128, 0), 1)
        cv2.putText(im, str(round(img_dist, 4)), (80, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        img = Image.fromarray(im).convert("RGBA")
        x = (i+1 - 1) % 5
        y = (i+1 - 1) // 5
        if y >10: print(y)
        bg.paste(img, (x * 150, y * 150))
    bg.show()

if __name__ == '__main__':
    # test for combine image
    list1 = ['/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose/query/0002_c002_00030600_0.jpg' for i in
             range(300)]
    list2 = [i+0.3121 for i in range(300)]
    zipped = list(zip(list1, list2))
    ui_result(zipped)