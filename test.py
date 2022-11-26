from scipy import ndimage 
import numpy as np
import math
import cv2
import random, string
import os


##########################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def lagrang(array):
    xk = [1, 3]
    result = []
    result.append(array[0])
    result.append(xk[0])
    result.append(array[1])
    result.append(xk[1])
    result.append(array[2])

    for k in range(len(xk)):
        cash1 = array[0] * ((xk[k]-2)*(xk[k]-4))/8
        cash2 = array[1] * (xk[k]*(xk[k]-4))/-4
        cash3 = array[2] * (xk[k]*(xk[k]-2))/8
        ck = cash1 + cash2 + cash3
        result[xk[k]] = math.floor(ck)
        if ck > 255:
            result[xk[k]] = 255
        if ck < 0:
            result[xk[k]] = 0
    return result

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def interpolation(array):
    # Переменные для увелечения по cтолбцу (5х5) -> 10х5
    rows = []  # для конечного массива
    block = []  # Место для хранения блоков 3х
    columnRowLen, columnColumnLen = np.shape(array)
    i = 0  # для итерации
    j = 0  # для итерации

    # Увелечили из 5х5 в 10х5
    while columnColumnLen > i:
        cashRow = []

        while columnRowLen > j:
            block.append(array[i][j])

            # print("i = ", i, "j = ", j)
            # print("block append", block)

            # calculate lagrang and clean block
            if len(block) == 5:

                firstPart = [block[0], block[1], block[2]]
                secondPart = [block[2], block[3], block[4]]

                # add to row result
                for k, value in enumerate(lagrang(firstPart)):
                    cashRow.append(value)

                # add to row result
                for k, value in enumerate(lagrang(secondPart)):
                    cashRow.append(value)

                block = []

            # clean read array
            j = j + 1

        rows.append(cashRow)
        cashRow = []
        j = 0
        i += 1

    # Переменные для увелечения по cтолбцу (10х5) -> 10х10
    columnRowLen, columnColumnLen = np.shape(rows)
    matrix = []  # финальная матрица
    block = []  # место для хранения
    i = 0  # для итерации
    j = 0  # для итерации

    # Увелечили из 5х5 в 10х5
    while columnColumnLen > i:
        cashRow = []

        columnValues = [x[i] for x in rows]  # это числа столбцов
        while len(columnValues) > j:
            block.append(rows[j][i])

            # print("i = ", i, "j = ", j)
            # print("block append", block)

            # calculate lagrang and clean block
            if len(block) == 5:
                # print("Block:", block)
                # print("\n")
                firstPart = [block[0], block[1], block[2]]
                secondPart = [block[2], block[3], block[4]]

                # add to final result
                for k, value in enumerate(lagrang(firstPart)):
                    cashRow.append(value)

                # add to final result
                for k, value in enumerate(lagrang(secondPart)):
                    cashRow.append(value)

                block = []

            # clean read array
            j = j + 1

        matrix.append(cashRow)
        cashRow = []
        j = 0
        i += 1

    rotated = ndimage.rotate(np.array(matrix), 270)
    im = np.fliplr(rotated)
    return im

def f(x):
    if x <= 1: return 1
    return f(x - 1) * x

def generateKey(x):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(x))    

def bitstring_to_bytes(s):
    return bytes(int(s[i : i + 8], 2) for i in range(0, len(s), 8))

def bytes_to_decimical(s):
    deg = [int(math.pow(2, i)) for i in range(0, 8)][::-1]
    num = 0
    for i in range(len(s)):
        num += deg[i] * int(s[i])
    return num

def decimical_to_bytes(s):
    b = ""
    while s:
        b += str(s % 2)
        s //= 2
    ans = ""
    for i in range(8-len(b)):
        ans += '0'
    ans += b[::-1]
    return ans

def toBinChar(x):
    tmp = [decimical_to_bytes(ord(x[i])) for i in range(0, len(x))] 
    ans = ""
    for i in tmp:
        ans += i
    return ans

def bezje(x):        
    n = 4
    t = np.linspace(0, 1, 20)
    Pt = []
    for step in t:
        i = 0
        cnt = 0
        while i <= n: 
            cnt += x[i] * (f(n) / (f(i) * f(n - i))) * ((1 - step) ** (n - i)) * (step ** i)
            i += 1
        Pt.append(cnt)
    return Pt

def encodeVernam(s, k):
    alpha = [chr(64+i) for i in range(1, 27)]
    x, y = [], []
    for i in range(len(s)):
        for j in range(0, 26):
            if alpha[j] == s[i]:
                x.append(j)
                break
        for j in range(0, 26):
            if alpha[j] == k[i]:
                y.append(j)
                break
    for i in range(len(x)):
        x[i] = (x[i]+y[i])%26
    x = [alpha[x[i]] for i in range(len(x))]
    y = ""
    for i in x:
        y += i
    return y

def decodeVernam(s, k):
    alpha = [chr(64+i) for i in range(1, 27)]
    x, y = [], []
    for i in range(len(s)):
        for j in range(0, 26):
            if alpha[j] == s[i]:
                x.append(j)
                break
        for j in range(0, 26):
            if alpha[j] == k[i]:
                y.append(j)
                break
    for i in range(len(x)):
        x[i] = (x[i]-y[i])%26
    x = [alpha[x[i]] for i in range(len(x))]
    y = ""
    for i in x:
        y += i
    return y

def encode(img, msgBin, file): 
    # переводим сообщения в двоичном формате
    msgBin = toBinChar(msgBin)

    # разбивание пикселей из пяти блоков 
    fiveBlock = []
    x = 0
    while x < len(img):
        px = []
        for i in range(len(img[x])):
            for j in range(3):
                px.append(img[x][i][j])
        l = 0
        r = 5
        c = 0
        n = len(px)
        tmp = []
        while r <= n:
            cpy = []
            for i in range(l, r):
                cpy.append(px[i])
            tmp.append(cpy)
            l = r
            r += 5
        fiveBlock.append(tmp)
        x += 1



    # fiveBlock = np.array(fiveBlock)
    # file.write("t = 0.05\n\n")
    # for i in range(len(fiveBlock[0])):
    #     file.write("block[" + str(i + 1) + "]" + " : " + str(fiveBlock[0][i]) + "\nBezje = " + str(bezje(fiveBlock[0][i])) + "\n\n")
        

    bit = 0  # указатель на текущий бит строки
    x = 0 # граница строки массива пикселей из пяти блоков 
    y = 0 # граница столбца массива пикселей из пяти блоков
    while True:
        
        # print(x, ' ', y, ' ', bit, end='\n')

        if x == len(fiveBlock): # если дошли до строки границы строки массива, выводим сообщение достигли края массива
            print("out of range!")
            break
        if y == len(fiveBlock[x]): # если дошли до границы столбца, начинаем с новой строки
            y = 0
            x += 1
        if bit == len(msgBin): # если указатель дошел до края бита строки, биты сообщения встроено!
            fiveBlock[x][y][0] = fiveBlock[x][y][2] = fiveBlock[x][y][4] = 0  # значения соседей точек P1, P3 зануляем как останочные значения для дешифровывание
            print("all message encoded!")
            break
      
        cnt = 0
        for i in range(1, len(fiveBlock[x][y])):
            if fiveBlock[x][y][i] == fiveBlock[x][y][0]:
                cnt += 1
            
        if cnt == len(fiveBlock[x][y]) - 1: # если все точки одинаковые, то пропускаем этот блок
            y += 1
            continue

        b = bezje(fiveBlock[x][y]) # вычисляем Безье из пяти точек (пикселей)


        if decimical_to_bytes(fiveBlock[x][y][1])[-1] == msgBin[bit]: # если младший бит точки P1 блока совпал с текущим битов строки 
            bit += 1  # то пропускаем точку P1
            if decimical_to_bytes(fiveBlock[x][y][3])[-1] == msgBin[bit]: # если младший бит точки P3 блока совпал с текущим битов строки 
                bit += 1 # то пропускаем точку P3
                y += 1 # пропускаем блок
            else: # иначе
                for e in b: # из Безье находим такое значение, что младший бит точки P3 совпал с текущим битом строки
                    if decimical_to_bytes(e)[-1] == msgBin[bit]:
                        fiveBlock[x][y][3] = e
                        bit += 1
                        y += 1
                        break
        else:
            for e in b: # из Безье находим такое значение, что младший бит точки P1 совпал с текущим битом строки
                if decimical_to_bytes(e)[-1] == msgBin[bit]:
                    fiveBlock[x][y][1] = e
                    bit += 1
                    break
            
            if decimical_to_bytes(fiveBlock[x][y][3])[-1] == msgBin[bit]: # если младший бит точки P3 блока совпал с текущим битов строки 
                bit += 1 # то пропускаем точку P3
                y += 1 # пропускаем блок
            else:
                for e in b: # из Безье находим такое значение, что младший бит точки P3 совпал с текущим битом строки
                    if decimical_to_bytes(e)[-1] == msgBin[bit]:
                        fiveBlock[x][y][3] = e
                        bit += 1
                        y += 1
                        break
            
 

    # массив пикселей из пяти блоков обратно переводим в формат массива картинки
    endImg = []
    z = 0
    while z < len(fiveBlock):
        px = []
        for i in range(len(fiveBlock[z])):
            for j in range(5):
                px.append(fiveBlock[z][i][j])
        
        l = 0
        r = 3
        c = 0
        n = len(px) 

        #this is Pointers

        tmp = [] 
        while r <= n:
            cpy = []
            for i in range(l, r):
                cpy.append(px[i])
            tmp.append(cpy)
            l = r
            r += 3

        endImg.append(tmp)
        z += 1

    

    # массив из пикселей обернем в массив
    endImg = np.array(endImg)

    # возращаем измененный массив пикселей картинки
    return endImg
    
def decode(img):
    # пустая строка для сбора битов 
    msgBin = ""
   
    # разбиение на пять блоков пикселей
    fiveBlock = []
    x = 0
    while x < len(img):
        px = []
        for i in range(len(img[x])):
            for j in range(3):
                px.append(img[x][i][j])
        l = 0
        r = 5
        c = 0
        n = len(px)
        tmp = []
        while r <= n:
            cpy = []
            for i in range(l, r):
                cpy.append(px[i])
            tmp.append(cpy)
            l = r
            r += 5
        fiveBlock.append(tmp)
        x += 1


    # собственно границы строки и столбцы массива из пяти блоков
    xx = 0 # граница строки массива из пяти блоков
    yy = 0 # граница столбца массива из пяти блоков
    while True: # работаем пока не достигнем резльутатов
        if xx == len(fiveBlock): # если граница строки достиг предела, останавливаем цикл
            break
        if yy == len(fiveBlock[xx]): # если граница столбца достиг предела, переходим в новую строку 
            yy = 0
            xx += 1
        if fiveBlock[xx][yy][0] == 0 and fiveBlock[xx][yy][2] == 0 and fiveBlock[xx][yy][4] == 0: # если соседние значения точки P1, P3 нули, то значить мы собрали все биты
            print("all message decoded!")
            break
    
        cnt = 0
        for i in range(1, len(fiveBlock[xx][yy])):
            if fiveBlock[xx][yy][i] == fiveBlock[xx][yy][0]:
                cnt += 1

        if cnt == len(fiveBlock[xx][yy]) - 1: # если значения пикселей одинаковы то, пропускаем текущий блок
            continue

        msgBin += decimical_to_bytes(fiveBlock[xx][yy][1])[-1] # кладем младший бит точки P1 в строку ответа
        msgBin += decimical_to_bytes(fiveBlock[xx][yy][3])[-1] # кладем младший бит точки P3 в строку ответа
        yy += 1
    
    # строку из двоичного формата переводим в байты символы
    return bitstring_to_bytes(msgBin)




if __name__ == "__main__":


    # INPUT DATA
    
    F1 = open("message.txt", "w")
    msg = generateKey(5000)    
    F1.write(msg)
    F2 = open("key.txt", "w")
    key = generateKey(len(msg))
    F2.write(key)
    encodedMsg = encodeVernam(msg, key)
    F3 = open("vernamEncode.txt", "w")
    F3.write(encodedMsg)



    # F1 = open("message.txt", "r+")
    # msg = F1.readline()
    # F2 = open("key.txt", "r+")
    # key = F2.readline()
    # F3 = open("vernamEncode.txt", "r+")
    # encodedMsg = F3.readline()
    # print("SECRET = {}\nKEY = {}\nENCODED MESSAGE = {}".format(msg, key, encodedMsg))




    # INTERPOLATION


    # image = cv2.imread("C:/Users/User01/Desktop/JOB/Stegonagraphy/225x225(input)/test17.png")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # grayImage = np.array(gray)
    # inter = np.array(interpolation(grayImage))    
    # cv2.imwrite("inter-img/inter764.BMP", inter)
    # print(cv2.imread("res-img/res1.png"))


    # collection = "test-img/"
    # for i, filename in enumerate(os.listdir(collection)):
    #     print("image[{}]".format(i + 1), end='\n')
    #     img  = cv2.imread(collection + filename)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     grayImg = np.array(gray)
    #     inter = np.array(interpolation(grayImg)) 
    #     cv2.imwrite("inter-img/inter" + str(i + 1) + ".BMP", inter)






    # ENCODE

    F4 = open("First_block_of_image082.txt", "w")
    img = cv2.imread("inter-img/inter82.BMP")
    ans = encode(img, encodedMsg, F4)
    # cv2.imwrite("res-img/res238.BMP", ans)


    # collection = "inter-img/"
    # for i, filename in enumerate(os.listdir(collection)): 
    #     if (i + 1) > 238:
    #         print("image[{}]".format(i + 1), end='\n')   
    #         img = cv2.imread("inter-img/" + filename)
    #         ans = encode(img, encodedMsg)
    #         cv2.imwrite("res-img/res" + str(i + 1) + ".BMP", ans)
        

        

    # DECODE

    # img = cv2.imread("res-img/res1.BMP")
    # decodedMsg = decode(img)
    # print(msg == decodeVernam(decodedMsg.decode("utf-8"), key))


    # collection = "res-img/"
    # for i, filename in enumerate(os.listdir(collection)): 
    #     if (i + 1) > 238:
    #         print("image[{}]".format(i + 1), end='\n')   
    #         img = cv2.imread("res-img/" + filename)
    #         ans = decode(img)
    #         print(msg == decodeVernam(decodedMsg.decode("utf-8"), key))




