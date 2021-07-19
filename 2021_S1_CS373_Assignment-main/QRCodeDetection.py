
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy
from pyzbar import pyzbar
from matplotlib.patches import Polygon
import tkinter
import tkinter.messagebox

import imageIO.png


#queue class
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)




def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)


    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()

def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(0.299*pixel_array_r[i][j] + 0.587*pixel_array_g[i][j] + 0.114*pixel_array_b[i][j])
    
    return greyscale_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    p_min, p_max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    greyscale_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] == p_min:
                greyscale_image[i][j] = 0
            elif pixel_array[i][j] == p_max:
                greyscale_image[i][j] = 255
            else:
                greyscale_image[i][j] = round(255*((pixel_array[i][j]-p_min)/(p_max-p_min)))
    return greyscale_image


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    p_min = 255
    p_max = 0
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < p_min:
                p_min = pixel_array[i][j]
            if pixel_array[i][j] > p_max:
                p_max = pixel_array[i][j]
    return p_min, p_max

def computeHorizontalEdgesSobel(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
        
    for i in range(image_height):
        for j in range(image_width):
            if i==0 or j==0 or i==image_height-1 or j==image_width-1:
                continue
            else:
                result[i][j] = (1/8)*((pixel_array[i-1][j-1]) + -1*(pixel_array[i+1][j-1]) + 2*(pixel_array[i-1][j]) + -2*(pixel_array[i+1][j]) +
                (pixel_array[i-1][j+1]) + -1*(pixel_array[i+1][j+1]))
    return result
    
def computeVerticalEdgesSobel(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
        
    for i in range(image_height):
        for j in range(image_width):
            if i==0 or j==0 or i==image_height-1 or j==image_width-1:
                continue
            else: #changed (1/8) to (-1/8) to invert it to follow the example
                result[i][j] = (-1/8)*(-1*(pixel_array[i-1][j-1]) + -2*(pixel_array[i][j-1]) + -1*(pixel_array[i+1][j-1]) + (pixel_array[i-1][j+1]) +
                2*(pixel_array[i][j+1]) + (pixel_array[i+1][j+1]))
    return result

def computeEdgeMagnitude(horizontal_edge, vertical_edge, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            result[i][j] = ((horizontal_edge[i][j])**2 + (vertical_edge[i][j])**2) ** (1/2)
    return result

def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    result = []
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        new_array[i] = [pixel_array[i][0]]+pixel_array[i]+[pixel_array[i][image_width-1]]
    new_array.insert(0, new_array[0])
    new_array.append(new_array[len(new_array)-1])

    
    for i in range(image_height):
        for j in range(image_width):
            if i==0 or j==0 or i==image_height-1 or j==image_width-1:
                result[i][j] = (1/16)*(new_array[i][j] + 2*new_array[i+1][j] + new_array[i+2][j] + 2*new_array[i][j+1] + 4*new_array[i+1][j+1] +
                2*new_array[i+2][j+1] + new_array[i][j+2] + 2*new_array[i+1][j+2] + new_array[i+2][j+2])
            else:
                result[i][j] = (1/16)*(pixel_array[i-1][j-1] + 2*pixel_array[i][j-1] + pixel_array[i+1][j-1] + 2*pixel_array[i-1][j] + 4*pixel_array[i][j] +
                2*pixel_array[i+1][j] + pixel_array[i-1][j+1] + 2*pixel_array[i][j+1] + pixel_array[i+1][j+1])
    return result

def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    result = []
    for i in range(image_height):
        result.append([0]*image_width)
        
        
    for i in range(image_height):
        for j in range(image_width):
            if i==0 or j==0 or i==image_height-1 or j==image_width-1:
                continue
            else:
                result[i][j] = abs((pixel_array[i-1][j-1] + pixel_array[i][j-1] + pixel_array[i+1][j-1] + pixel_array[i-1][j] + pixel_array[i][j] +
                pixel_array[i+1][j] + pixel_array[i-1][j+1] + pixel_array[i][j+1] + pixel_array[i+1][j+1])/9)
    return result

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold_value:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if i == 0 or j == 0 or i == image_height-1 or j == image_width-1:
                continue
            else:
                if (pixel_array[i-1][j-1] >= 1) and (pixel_array[i][j-1] >= 1) and (pixel_array[i+1][j-1] >= 1) and (pixel_array[i-1][j] >= 1) and (pixel_array[i][j] >= 1) and (pixel_array[i+1][j] >= 1) and (pixel_array[i-1][j+1] >= 1) and (pixel_array[i][j+1] >= 1) and (pixel_array[i+1][j+1] >= 1):
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    return result

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    if pixel_array[0][0] >= 1: #top-left corner
        result[0][0] = 1
        result[0][1] = 1
        result[1][0] = 1
        result[1][1] = 1
    if pixel_array[0][image_width-1] >= 1: #top-right corner
        result[0][image_width-1] = 1
        result[0][image_width-2] = 1
        result[1][image_width-1] = 1
        result[1][image_width-2] = 1
    if pixel_array[image_height-1][0] >= 1: #bottom-left corner
        result[image_height-1][0] = 1
        result[image_height-1][1] = 1
        result[image_height-2][0] = 1
        result[image_height-2][1] = 1
    if pixel_array[image_height-1][image_width-1] >= 1: #bottom-right corner
        result[image_height-1][image_width-1] = 1
        result[image_height-1][image_width-2] = 1
        result[image_height-2][image_width-1] = 1
        result[image_height-2][image_width-2] = 1
    for column in range(1, image_width-1): #first row
        if pixel_array[0][column] >= 1:
            result[0][column] = 1
            result[0][column-1] = 1
            result[0][column+1] = 1
            result[1][column] = 1
            result[1][column-1] = 1
            result[1][column+1] = 1
    for column in range(1, image_width-1): #last row
        if pixel_array[image_height-1][column] >= 1:
            result[image_height-1][column] = 1
            result[image_height-1][column-1] = 1
            result[image_height-1][column+1] = 1
            result[image_height-2][column] = 1
            result[image_height-2][column-1] = 1
            result[image_height-2][column+1] = 1
    for row in range(1, image_height-1): #first column
        if pixel_array[row][0] >= 1:
            result[row][0] = 1
            result[row-1][0] = 1
            result[row+1][0] = 1
            result[row][1] = 1
            result[row-1][1] = 1
            result[row+1][1] = 1
    for row in range(1, image_height-1): #last column
        if pixel_array[row][image_width-1] >= 1:
            result[row][image_width-1] = 1
            result[row-1][image_width-1] = 1
            result[row+1][image_width+1] = 1
            result[row][image_width-2] = 1
            result[row-1][image_width-2] = 1
            result[row+1][image_width-2] = 1

    for i in range(1, image_height-1): #pixels in the middle
        for j in range(1, image_width-1):
            if pixel_array[i][j] >= 1:
                result[i-1][j-1] = 1
                result[i][j-1] = 1
                result[i+1][j-1] = 1
                result[i-1][j] = 1
                result[i][j] = 1
                result[i+1][j] = 1
                result[i-1][j+1] = 1
                result[i][j+1] = 1
                result[i+1][j+1] = 1
    return result

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height) #not visited: 0, visited: 1
    
    current_label = 1 
    pixel_size = {}
    
    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] > 0) and (visited[i][j] == 0): #if f(j,i) is object and (j,i) is not visited
                q = Queue()
                q.enqueue((j, i))
                visited[i][j] = 1
                total_pixel = 0
                while q.isEmpty() == False:
                    (j1, i1) = q.dequeue()
                    result[i1][j1] = current_label
                    total_pixel += 1
                    if (j1-1 >= 0) and (pixel_array[i1][j1-1] > 0) and (visited[i1][j1-1] == 0): #left neighbour
                        q.enqueue((j1-1, i1))
                        visited[i1][j1-1] = 1
                    if (j1+1 <= image_width-1) and (pixel_array[i1][j1+1] > 0) and (visited[i1][j1+1] == 0): #right neighbour
                        q.enqueue((j1+1, i1))
                        visited[i1][j1+1] = 1
                    if (i1-1 >= 0) and (pixel_array[i1-1][j1] > 0) and (visited[i1-1][j1] == 0): #upper neighbour
                        q.enqueue((j1, i1-1))
                        visited[i1-1][j1] = 1
                    if (i1+1 <= image_height-1) and ((pixel_array[i1+1][j1] > 0) and (visited[i1+1][j1] == 0)): #lower neighbour
                        q.enqueue((j1, i1+1))
                        visited[i1+1][j1] = 1
                pixel_size[current_label] = total_pixel
                current_label += 1
    
    return (result, pixel_size)

def cropImage(image, new_image, y_start, height, x_start, width): #crop out the qr code part from the original image, and save it as a new PNG file
    img = pyplot.imread(image)[y_start:height, x_start:width]
    #cropped = img[y_start-abs(height):y_start, x_start:x_start+abs(width)]
    #img = numpy.dot(img[...,:3], [0.299, 0.587, 0.114])
    pyplot.imsave(new_image, img)

def drawBoundingBox(pixel_array, image_height, image_width):
    is_first = True
    x_diff = 10
    result = []
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] != 0:
                if is_first == True: # top pixel
                    is_first = False
                    top = (j, i)
                    min_height = i
                    max_height = i
                    min_width = j
                    max_width = j
                    top_x, top_y = j, i
                if j > max_width: # right most pixel
                    max_width = j
                    right_x, right_y = j, i
                if i > max_height: # bottom pixel
                    max_height = i
                    bottom_x, bottom_y = j, i
                if j < min_width: # left most pixel
                    min_width = j
                    left_x, left_y = j, i
  
    if abs(top_x - left_x) >= x_diff: # recognize a rotated image by checking if the left most pixel and the top pixel have the same x-value
        print("QR code is rotated")
        rect = Polygon([(top_x,top_y),(right_x,right_y),(bottom_x,bottom_y),(left_x,left_y)], linewidth = 3, edgecolor = 'g', facecolor = 'none')
        rotated = True
        result.extend((rect, (top_x,top_y), (right_x,right_y), (bottom_x,bottom_y), (left_x,left_y), rotated))
    else:
        print("QR code is not rotated")
        rect = Polygon([(min_width,min_height),(max_width,min_height),(max_width,max_height),(min_width,max_height)], linewidth = 3, edgecolor = 'g', facecolor = 'none')
        rotated = False
        result.extend((rect, (min_width,min_height), (max_width,min_height), (max_width,max_height), (min_width,max_height), rotated))
    return result




def main():
    #filename = "./images/covid19QRCode/poster1small.png" #displays bounding box correctly, can decode the content of the QR code
    #filename = "./images/covid19QRCode/challenging/bch.png" #displays bounding box correctly, can decode the content of the QR code
    #filename = "./images/covid19QRCode/challenging/connecticut.png" #displays bounding box correctly, can decode the content of the QR code
    #filename = "./images/covid19QRCode/challenging/bloomfield.png" #displays bounding box correctly, cannot decode the content of the QR code
    #filename = "./images/covid19QRCode/challenging/playground.png" #cannot display the bounding box correctly, cannot decode the content of the QR code
    #filename = "./images/covid19QRCode/challenging/poster1smallrotated.png" #displays bounding box correctly, can decode the content of the QR code
    filename = "./images/covid19QRCode/challenging/shanghai.png" #displays bounding box correctly, can decode the content of the QR code

    new_image = "./images/covid19QRCode/new_image.png"

    greyscale_img = "./images/covid19QRCode/greyscale.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)

    #----------------------------Step 1: convert the image into grayscale, and stretch it to be between 0 and 255-------------------------------------
    print("Step 1: converting the image into greyscale...")
    greyscale_image = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    greyscale_stretched = scaleTo0And255AndQuantize(greyscale_image, image_width, image_height)

    writeGreyscalePixelArraytoPNG(greyscale_img, greyscale_stretched, image_width, image_height)

    #----------------------------Step 2: 3x3 sobel filter mask used to compute horizontal edge--------------------------------------------------------
    print("Step 2: computing horizontal edges sobel...")
    horizontal_edge = computeHorizontalEdgesSobel(greyscale_stretched, image_width, image_height)

    #----------------------------Step 3: 3x3 sobel filter mask used to compute vertical edge----------------------------------------------------------
    print("Step 3: computing veritical edges sobel...")
    vertical_edge = computeVerticalEdgesSobel(greyscale_stretched, image_width, image_height) 

    #----------------------------Step 4: compute the magnitude----------------------------------------------------------------------------------------
    print("Step 4: computing edge magnitude...")
    edge_magnitude = computeEdgeMagnitude(horizontal_edge, vertical_edge, image_width, image_height)

    #------------------------Step 5: smooth using the mean filter 11 times to get the right output, and then stretch it to be between 0 and 255-------
    print("Step 5: smoothing the image...")
    smoothed = computeBoxAveraging3x3(edge_magnitude, image_width, image_height)

    for i in range(10):
        smoothed = computeBoxAveraging3x3(smoothed, image_width, image_height)
    smoothed_stretched = scaleTo0And255AndQuantize(smoothed, image_width, image_height)

    #----------------------------Step 6: get the edge regions as a binary image using threshold-------------------------------------------------------
    print("Step 6: computing edge regions as a binary image...")
    edge_binary = computeThresholdGE(smoothed_stretched, 70, image_width, image_height)

    #----------------------------Step 7: perform a closing operation----------------------------------------------------------------------------------
    print("Step 7: performing the closing operation...")
    dilation = computeDilation8Nbh3x3FlatSE(edge_binary, image_width, image_height)

    for i in range(1):
        dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)
    
    erosion = computeErosion8Nbh3x3FlatSE(dilation, image_width, image_height)

    #----------------------------Step 8: find the largest connected component-------------------------------------------------------------------------
    print("Step 8: finding the largest connected component...")
    (connected, pixel_size) = computeConnectedComponentLabeling(erosion, image_width, image_height)

    max_sized_label = 1
    max_size = 0
    for key, item in pixel_size.items():
        if item > max_size:
            max_size = item
            max_sized_label = key
    
    for i in range(image_height):
        for j in range(image_width):
            if connected[i][j] != max_sized_label:
                connected[i][j] = 0

    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))
    #pyplot.imshow(connected, cmap="gray")
 

    # get access to the current pyplot figure
    axes = pyplot.gca()


    #----------------------------Step 9: draw the bounding box----------------------------------------------------------------------------------------
    print("Step 9: drawing the bounding box...")
    bounding_box_list = drawBoundingBox(connected, image_height, image_width)
    rect = bounding_box_list[0]
    # paint the rectangle over the current plot
    axes.add_patch(rect)

    content_list = []
    if bounding_box_list[5] == False:
        cropImage(greyscale_img, new_image, bounding_box_list[1][1], bounding_box_list[3][1], bounding_box_list[1][0], bounding_box_list[2][0])
    else:
        cropImage(greyscale_img, new_image, bounding_box_list[1][1], bounding_box_list[3][1], bounding_box_list[4][0], bounding_box_list[2][0])
    image_to_decode = pyplot.imread(new_image)
    image_to_decode = image_to_decode.astype("float") * 255.0
    result = pyzbar.decode(image_to_decode)
    for item in result:
        item_data = item.data.decode("UTF-8")
        content_list.append(item_data)
        print("{}".format(item_data))
    content_string = "".join(content_list)
    if len(content_list) < 1:
        tkinter.messagebox.showinfo("QR Code content", "Cannot decode the QR code. \n\nClick on OK to view the image and the bounding box.")
    else:
        tkinter.messagebox.showinfo("QR Code content", "{} \n\nClick on OK to view the image and the bounding box.".format(content_string))

    # plot the current figure
    pyplot.show()




if __name__ == "__main__":
    main()




