import tkinter
from PIL import Image
from os import path
from tkinter import messagebox
from PIL import ImageTk
import cv2
import matplotlib.pyplot as plt
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def test_function(button_name="don't know"):
    print("This is a test function for button", button_name)


# menu label class (Like filter menu)
class MenuLabel(tkinter.Label):
    def __init__(self, master, text_ref):
        tkinter.Label.__init__(self, master=master)
        # super().__init__()  # in the brackets of super RedFrame, self
        self["font"] = 'device 18'
        self["text"] = text_ref
        self["height"] = 1
        self["width"] = 27
        self["relief"] = tkinter.RIDGE
        self["bd"] = 3
        self["bg"] = '#ED7D7D'
        self["fg"] = '#103548'


# Filter button class
class FilterButton(tkinter.Button):
    def __init__(self, master, text, function_ref, row_ref, column_ref, function_argument="Don't know"):
        tkinter.Button.__init__(self, master=master)
        # super().__init__()  # in the brackets of super RedFrame, self
        self["font"] = 'courier 12'
        self["text"] = text
        self["height"] = 1
        self["width"] = 12
        self["relief"] = tkinter.RIDGE
        self["bd"] = 3
        self["bg"] = '#ad458f'
        self["fg"] = '#ffaaff'
        self["padx"] = 0
        self["command"] = lambda: function_ref(function_argument)
        self.grid(row=row_ref, column=column_ref)


# Frame for buttons
class InputFrame(tkinter.Frame):
    def __init__(self, master, height_ref, bg_ref, row_ref):
        tkinter.Frame.__init__(self, master=master)
        # super().__init__()  # in the brackets of super RedFrame, self
        self["width"] = 400
        self["height"] = height_ref
        self["bg"] = bg_ref
        self["bd"] = 3
        self["relief"] = 'groove'
        self.grid(row=row_ref, column=0, sticky=tkinter.W)


# Label for crop
class CropLabel(tkinter.Label):
    def __init__(self, master, text_ref, row_ref, column_ref):
        tkinter.Label.__init__(self, master=master)
        # super().__init__()  # in the brackets of super RedFrame, self
        self["text"] = text_ref
        self["font"] = 'comic 8'
        self['width'] = 12
        self["relief"] = tkinter.RIDGE
        self["bd"] = 3
        self["bg"] = '#F76DDE'
        self["fg"] = '#03002C'
        self["padx"] = 0
        self.grid(row=row_ref, column=column_ref)


# add the photo to the input frame
def display_photo_input(photo_path):
    img_tk = ImageTk.PhotoImage(Image.open(photo_path))
    panel_input.configure(image=img_tk, height=360, width=880)
    panel_input.image = img_tk


# add the photo to the output frame
def display_photo_output(photo_path='output.png'):
    ready_photo(photo_path, photo_path)
    img_tk = ImageTk.PhotoImage(Image.open(photo_path))
    panel_output.configure(image=img_tk, height=360, width=880)
    panel_output.image = img_tk


def resize_photo(photo_temp):
    width, height = photo_temp.size
    new_width, new_height = width, height
    if width > 880:
        new_width = 880
        multiplier = new_width/width
        new_height = int(height*multiplier)
    if height > 360:
        new_height = 360
        multiplier = new_height/height
        new_width = int(width*multiplier)
    photo_final = photo_temp.resize((new_width, new_height), Image.ANTIALIAS)

    return photo_final


def ready_photo(photo_path, photo_name):
    photo_temp = Image.open(photo_path)
    photo_temp = resize_photo(photo_temp)
    photo_temp.save(photo_name)


def path_checker():
    file_path = entry_path.get()
    print(file_path)
    if path.exists(file_path):
        ready_photo(file_path, 'input_display.png')
        display_photo_input('input_display.png')
        resize(file_path)
    else:
        messagebox.showerror('ERROR!', 'the input path is incorrect')
        var_entry_path.set('')



# Definition of of xrange
def xrange(x):
    return iter(range(x))


# takes path input & save the photo at end
# resize photo to make it workable in cv
def resize(img_path):
    resize_img = cv2.imread(img_path)
    height, width = resize_img.shape[:2]

    if width > 1366:
        mul = 1366 / width
        height = height * mul
        width = width * mul
    if height > 768:
        mul = 768 / height
        height = height * mul
        width = width * mul
    width = int(width)
    height = int(height)

    sized_img = cv2.resize(resize_img, (width, height))
    cv2.imwrite('resize.jpg', sized_img)


# take image path as input & save the photo to display
# function to cartooning the image
def cartooning(image_path='resize.jpg'):
    img_rgb = cv2.imread(image_path)
    numDownSamples = 2  # number of downscaling steps
    numBilateralFilters = 50  # number of bilateral filtering steps

    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in xrange(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in xrange(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 5)

    # upsample image to original size
    for _ in xrange(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 9)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed with color image
    (x, y, z) = img_color.shape
    img_edge = cv2.resize(img_edge, (y, x))
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    tmp_canvas = cv2.bitwise_and(img_color, img_edge)
    cv2.imwrite("output.png", tmp_canvas)
    display_photo_output("output.png")


# Function to make image black and white
def to_black(image_path='resize.jpg'):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output.png", gray_img)
    display_photo_output("output.png")


# Function to enhance the image
def enhance(image_path='resize.jpg'):
    img = cv2.imread(image_path)

    # Preparation for CLAHE
    clahe = cv2.createCLAHE()

    # convert to gray scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply enhancement
    enhance_img = clahe.apply(gray_img)
    cv2.imwrite("output.png", enhance_img)
    display_photo_output("output.png")


# Function to create a sketch of image
def to_sketch(sketch_scale=3, image_path='resize.jpg'):
    img = cv2.imread(image_path, 0)

    if is_number(sketch_scale):
        sketch_scale = int(sketch_scale)
    else:
        sketch_scale = 3

    if sketch_scale < 3:
        sketch_scale = 3
    elif sketch_scale > 9:
        sketch_scale = 9
    blur = cv2.medianBlur(img, sketch_scale)
    sketch = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    cv2.imwrite("output.png", sketch)
    display_photo_output("output.png")


# Function for binary filter
def to_binary(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    th = 127
    max_val = 255

    ret, binary_img = cv2.threshold(img, th, max_val, cv2.THRESH_BINARY)
    cv2.imwrite("output.png", binary_img)
    display_photo_output("output.png")


# Function to Blur the image
def to_blur(blur_scale=3, image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    if is_number(blur_scale):
        blur_scale = int(blur_scale)
    else:
        blur_scale = 3
    if blur_scale < 3:
        blur_scale = 3
    elif blur_scale > 25:
        blur_scale = 25

    # print(blur_scale, type(blur_scale))
    blur_img = cv2.medianBlur(img, blur_scale)

    cv2.imwrite("output.png", blur_img)
    display_photo_output("output.png")


# Function to convert BGR to RGB
def to_salt_pepper(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite("output.png", converted_img)
    display_photo_output("output.png")


# Function to convert BGR to XYZ
def to_salt_touch(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    cv2.imwrite("output.png", converted_img)
    display_photo_output("output.png")


# Function convert BGR to RGB and clahe
def to_clahe(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    output2_R = clahe.apply(R)
    output2_G = clahe.apply(G)
    output2_B = clahe.apply(B)

    clahe_filter = cv2.merge((output2_R, output2_G, output2_B))
    cv2.imwrite("output.png", clahe_filter)
    display_photo_output("output.png")


def mirror(mirror_choice, image_path='resize.jpg'):
    print("Mirror the image")

    img = cv2.imread(image_path)

    print("1. mirror vertically")
    print("2. mirror horizontally")

    if mirror_choice == 1:
        img_flip = cv2.flip(img, 0)
    elif mirror_choice ==2:
        img_flip = cv2.flip(img, 1)

    cv2.imwrite("output.png", img_flip)
    display_photo_output("output.png")


def rotate(rotate_choice, image_path='resize.jpg'):
    print("Rotate image")

    img = cv2.imread(image_path)

    print("1. Rotate 90 degree clockwise")
    print("2. Rotate 180 degree clockwise")
    print("3. Rotate 270 degree clockwise")

    if rotate_choice == 1:
        img_transpose = cv2.transpose(img)
        img_flip = cv2.flip(img_transpose, 1)
    elif rotate_choice == 2:
        img_flip = cv2.flip(img, 0)
    elif rotate_choice == 3:
        img_transpose = cv2.transpose(img)
        img_flip = cv2.flip(img_transpose, 0)

    cv2.imwrite("output.png", img_flip)
    display_photo_output("output.png")


def user_resize(resize_choice, new_height=0, new_width=0, image_path='resize.jpg'):
    print("Resize the image")

    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]

    if is_number(new_height):
        new_height = int(new_height)
    else:
        new_height = original_height

    if is_number(new_width):
        new_width = int(new_width)
    else:
        new_width = original_width

    if resize_choice == 1:
        height, width = img.shape[:2]
        print("The height and weight of the image is", height, "&", width)

        resize_img = cv2.resize(img, (new_width, new_height))
    elif resize_choice == 2:  # based on height
        height, width = img.shape[:2]
        print("The height and weight of the image is", height, "&", width)

        multiplier = new_height / height
        new_width = multiplier * width
        new_width = int(new_width)

        resize_img = cv2.resize(img, (new_width, new_height))
    elif resize_choice == 3:  # based on width
        height, width = img.shape[:2]
        print("The height and weight of the image is", height, "&", width)

        multiplier = new_width / width
        new_height = multiplier * height
        new_height = int(new_height)

        resize_img = cv2.resize(img, (new_width, new_height))

    cv2.imwrite("output.png", resize_img)
    display_photo_output("output.png")


def crop(top_left_x, top_left_y, bottom_right_x, bottom_right_y, image_path='resize.jpg'):
    print("Crop image")

    img = cv2.imread(image_path)

    height, width = img.shape[:2]

    if is_number(top_left_x):
        top_left_x = int(top_left_x)
    else:
        top_left_x = 0

    if is_number(top_left_y):
        top_left_y = int(top_left_y)
    else:
        top_left_y = 0

    if is_number(bottom_right_x):
        bottom_right_x = int(bottom_right_x)
    else:
        bottom_right_x = width

    if is_number(bottom_right_y):
        bottom_right_y = int(bottom_right_y)
    else:
        bottom_right_y = height

    print("The height and weight of the image is", height, "&", width)

    crop_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    cv2.imwrite("output.png", crop_img)
    display_photo_output("output.png")


def numpy_histogram(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])

    plot_temp_save = 'numpy plot.png'

    plt.subplot(1, 2, 2)
    hist, bins = np.histogram(img.ravel(), 256, [0, 255])
    plt.plot(hist)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.5)
    plt.savefig(plot_temp_save)
    cv2_plot_img = cv2.imread(plot_temp_save)
    cv2.imshow("Numpy Histogram", cv2_plot_img)


def rgb_histogram(image_path='resize.jpg'):
    img = cv2.imread(image_path, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(img)
    plt.subplot(3, 1, 1)
    hist, bins = np.histogram(R.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist, color='r')
    plt.title('Red Histogram')

    plt.subplot(3, 1, 2)
    hist, bins = np.histogram(G.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist, color='g')
    plt.title('Green Histogram')

    plt.subplot(3, 1, 3)
    hist, bins = np.histogram(B.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist, color='b')
    plt.title('Blue Histogram')

    # gives space between two plot
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)

    plot_save_name = "RGB histogram.jpg"

    plt.savefig(plot_save_name)
    rgb_image = cv2.imread(plot_save_name)
    cv2.imshow("RGB histogram", rgb_image)


def save(img_path='output.png', output_name="final_Output.jpg"):

    if output_name == "":
        output_name = 'final_Output.jpg'

    img_save = cv2.imread(img_path)
    cv2.imwrite(output_name, img_save)
    messagebox.showinfo("Saved", "The photo is successfully saved")


window = tkinter.Tk()

window.iconbitmap('logo_1.ico')
window.title("360 Image editor")

font_label = 'fixedsys 16'
var_entry_path = tkinter.StringVar()

window.geometry('1288x720+120+50')   # another argument '1288x720+30+0'
window.resizable(width=False, height=False)

frame_main_image = tkinter.Frame(window, width=880, height=720)
frame_main_button = tkinter.Frame(window, width=400, height=720)

frame_main_image.grid(row=0, column=0)
frame_main_button.grid(row=0, column=1, sticky=tkinter.W)

# frame height input + filter = 36 + 144

# create sub frames
frame_photo_1 = tkinter.Frame(frame_main_image, width=880, height=360, bg='#ff0000', bd=3, relief='sunken')
frame_photo_2 = tkinter.Frame(frame_main_image, width=880, height=360, bg='#00ff00', bd=3, relief='sunken')

# attaching the 2 photo sub frame
frame_photo_1.grid(row=0, column=0)
frame_photo_2.grid(row=1, column=0)


# all edit frames
frame_all_info = InputFrame(frame_main_button, 36, '#ff4785', 0)
frame_input = InputFrame(frame_main_button, 36, '#ffff00', 1)
frame_filter_1 = InputFrame(frame_main_button, 144, '#0000ff', 2)
frame_filter_2 = InputFrame(frame_main_button, 72, '#ff00ff', 3)
frame_tools_1 = InputFrame(frame_main_button, 72, '#1a34a3', 4)
frame_tools_2 = InputFrame(frame_main_button, 36, '#2334f4', 5)
frame_tools_3 = InputFrame(frame_main_button, 164, '#236743', 6)
frame_tools_4 = InputFrame(frame_main_button, 108, '#23f743', 7)
frame_info = InputFrame(frame_main_button, 72, '#a3f7f3', 8)
frame_save = InputFrame(frame_main_button, 36, '#a457f3', 9)

# photo info shower
var_info_main = tkinter.StringVar()
label_info_main = tkinter.Label(frame_all_info, fg='#478491',
                                font="arial 12", width=42,
                                textvariable=var_info_main)
var_info_main.set("Nothing to show here now!")
label_info_main.pack()

# path label
label_path = tkinter.Label(frame_input, text="Path", fg='blue', font=font_label, anchor=tkinter.NW, padx=3)
label_path.grid(row=0, column=0)

# path entry
entry_path = tkinter.Entry(frame_input, width=45, textvariable=var_entry_path)
entry_path.grid(row=0, column=1)

# path entry button
button_path_add = tkinter.Button(frame_input, text='Open', padx=14,
                                 command=path_checker)
button_path_add.grid(row=0, column=2)

# input image when nothing is there
img_input = ImageTk.PhotoImage(Image.open('input_logo.png'))
panel_input = tkinter.Label(frame_photo_1, image=img_input)
panel_input.pack(side="bottom", fill="both", expand="yes")

# output image when nothing is there
img_output = ImageTk.PhotoImage(Image.open('output_logo.png'))
panel_output = tkinter.Label(frame_photo_2, image=img_output)
panel_output.pack(side="bottom", fill="both", expand="yes")

label_menu = MenuLabel(frame_filter_1, 'Filter Menu')
label_menu.grid(row=2, column=0, columnspan=3)


button_1 = tkinter.Button(frame_filter_1, text="Cartooning",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: cartooning()).grid(row=3, column=0)
button_2 = tkinter.Button(frame_filter_1, text="Soft Touch",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: to_salt_touch()).grid(row=3, column=1)
button_3 = tkinter.Button(frame_filter_1, text="Enhance",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: enhance()).grid(row=3, column=2)
button_4 = tkinter.Button(frame_filter_1, text="Clance",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: to_clahe()).grid(row=4, column=0)
button_5 = tkinter.Button(frame_filter_1, text="Salt& Paper",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: to_salt_pepper()).grid(row=4, column=1)
button_6 = tkinter.Button(frame_filter_1, text="black & White",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: to_black()).grid(row=4, column=2)
button_7 = tkinter.Button(frame_filter_1, text="Binary Filter",
                          font='courier 12', height=1, width=12,
                          relief=tkinter.RIDGE, bd=3,
                          bg='#ad458f', fg='#ffaaff', padx=0,
                          command=lambda: to_binary()).grid(row=5, column=1)

# create the blur panel
var_blur = tkinter.StringVar()

label_blur = tkinter.Label(frame_filter_2, text="Blur Strength",
                           fg='blue', font='fixedsys 16',
                           bd=3, relief='raised',
                           width=13, height=1, padx=3)
label_blur.grid(row=0, column=0)

# blur sketch entry
entry_blur = tkinter.Entry(frame_filter_2, width=30, textvariable=var_blur)
entry_blur.grid(row=0, column=1)

# blur sketch entry button
button_blur = tkinter.Button(frame_filter_2, text='Blur',
                             padx=2, font='fixedsys 12',
                             width=7, height=1,
                             fg='#5477fd', command=lambda: to_blur(var_blur.get()))
button_blur.grid(row=0, column=2)


# create the sketch panel
var_sketch = tkinter.StringVar()

label_sketch = tkinter.Label(frame_filter_2, text="Sketch Strength",
                             fg='blue', font='fixedsys 12',
                             width=15, height=1,
                             bd=3, relief='raised', padx=3)
label_sketch.grid(row=1, column=0)

# sketch strength entry
entry_sketch = tkinter.Entry(frame_filter_2, width=30, textvariable=var_sketch)
entry_sketch.grid(row=1, column=1)

# sketch strength button
button_sketch = tkinter.Button(frame_filter_2, text='Sketch',
                               padx=2, font='fixedsys 8',
                               width=7, height=1,
                               fg='#5477fd', command=lambda: to_sketch(var_sketch.get()))
button_sketch.grid(row=1, column=2)

# creating tools menu label
label_menu_tools = MenuLabel(frame_tools_1, 'Tools Menu')
label_menu_tools.grid(row=0, column=0, columnspan=2)

# create Mirror Button
button_vertical = tkinter.Button(frame_tools_1, text="Vertically Mirror",
                                 font='courier 12', height=1, width=18,
                                 relief=tkinter.RIDGE, bd=3,
                                 bg='#ad458f', fg='#ffaaff', padx=0,
                                 command=lambda: mirror(1)).grid(row=1, column=0)
button_horizontal = tkinter.Button(frame_tools_1, text="Horizontally Mirror",
                                   font='courier 12', height=1, width=19,
                                   relief=tkinter.RIDGE, bd=3,
                                   bg='#ad458f', fg='#ffaaff', padx=0,
                                   command=lambda: mirror(2)).grid(row=1, column=1)


# Rotate buttons
button_tools_rotate_90 = tkinter.Button(frame_tools_2, text="Rotate 90",
                                        font='courier 12', height=1, width=12,
                                        relief=tkinter.RIDGE, bd=3,
                                        bg='#ad458f', fg='#ffaaff', padx=0,
                                        command=lambda: rotate(1)).grid(row=1, column=0)
button_tools_rotate_180 = tkinter.Button(frame_tools_2, text="Rotate 180",
                                         font='courier 12', height=1, width=12,
                                         relief=tkinter.RIDGE, bd=3,
                                         bg='#ad458f', fg='#ffaaff', padx=0,
                                         command=lambda: rotate(2)).grid(row=1, column=1)
button_tools_rotate_270 = tkinter.Button(frame_tools_2, text="Rotate 270",
                                         font='courier 12', height=1, width=12,
                                         relief=tkinter.RIDGE, bd=3,
                                         bg='#ad458f', fg='#ffaaff', padx=0,
                                         command=lambda: rotate(3)).grid(row=1, column=2)


# create free resize entry panel
var_resize_width = tkinter.StringVar()
var_resize_height = tkinter.StringVar()

# free width label
label_resize_width_free = tkinter.Label(frame_tools_3, text="Width",
                                        fg='blue', font='fixedsys 16',
                                        bd=3, relief='raised',
                                        width=13, height=1, padx=3)
label_resize_width_free.grid(row=0, column=0)
# free height label
label_resize_height_free = tkinter.Label(frame_tools_3, text="height",
                                         fg='blue', font='fixedsys 16',
                                         bd=3, relief='raised',
                                         width=13, height=1, padx=3)
label_resize_height_free.grid(row=1, column=0)


# free width entry
entry_resize_width_free = tkinter.Entry(frame_tools_3, width=30, textvariable=var_resize_width)
entry_resize_width_free.grid(row=0, column=1)
# height entry
entry_resize_height_free = tkinter.Entry(frame_tools_3, width=30, textvariable=var_resize_height)
entry_resize_height_free.grid(row=1, column=1)

# free height entry button
button_resize_free = tkinter.Button(frame_tools_3, text='Resize',
                                    padx=2, font='fixedsys 12',
                                    width=7, height=2,
                                    fg='#5477fd', bg="#65F7D1",
                                    command=lambda: user_resize(1, var_resize_height.get(), var_resize_width.get()))
button_resize_free.grid(row=0, column=2, rowspan=2)

# width based resize
var_resize_width_fixed = tkinter.StringVar()
# width label
label_resize_width = tkinter.Label(frame_tools_3, text="Width",
                                   fg='blue', font='fixedsys 16',
                                   bd=3, relief='raised',
                                   width=13, height=1, padx=3)
label_resize_width.grid(row=2, column=0)
# width entry button
entry_resize_width = tkinter.Entry(frame_tools_3, width=30, textvariable=var_resize_width_fixed)
entry_resize_width.grid(row=2, column=1)
# width resize button
button_width_resize = tkinter.Button(frame_tools_3, text='Resize',
                                     padx=2, font='fixedsys 12',
                                     width=7, height=1,
                                     fg='#5477fd', bg="#65F7D1",
                                     command=lambda: user_resize(3, 'nothing', var_resize_width_fixed.get()))
button_width_resize.grid(row=2, column=2)


# height based resize
var_resize_height_fixed = tkinter.StringVar()
# height label
label_resize_height = tkinter.Label(frame_tools_3, text="Height",
                                    fg='blue', font='fixedsys 16',
                                    bd=3, relief='raised',
                                    width=13, height=1, padx=3)
label_resize_height.grid(row=3, column=0)
# height entry button
entry_resize_height = tkinter.Entry(frame_tools_3, width=30, textvariable=var_resize_height_fixed)
entry_resize_height.grid(row=3, column=1)
# height resize button
button_height_resize = tkinter.Button(frame_tools_3, text='Resize',
                                      padx=2, font='fixedsys 12',
                                      width=7, height=1,
                                      fg='#5477fd', bg="#65F7D1",
                                      command=lambda: user_resize(2, var_resize_height_fixed.get(), 'nothing'))
button_height_resize.grid(row=3, column=2)

# crop label shower
label_crop = MenuLabel(frame_tools_4, "Crop Menu")
label_crop.grid(row=0, column=0, columnspan=5)

# label for crop
label_top_x = CropLabel(frame_tools_4, "Top Left x", 1, 0)
label_top_y = CropLabel(frame_tools_4, "Top Left y", 1, 2)
label_bottom_x = CropLabel(frame_tools_4, "Bottom right x", 2, 0)
label_bottom_y = CropLabel(frame_tools_4, "Bottom right y", 2, 2)

# text variable for crop
var_top_x = tkinter.StringVar()
var_top_y = tkinter.StringVar()
var_bottom_x = tkinter.StringVar()
var_bottom_y = tkinter.StringVar()

# entry for crop
entry_top_x = tkinter.Entry(frame_tools_4, width=10, textvariable=var_top_x).grid(row=1, column=1)
entry_top_y = tkinter.Entry(frame_tools_4, width=10, textvariable=var_top_y).grid(row=1, column=3)
entry_bottom_x = tkinter.Entry(frame_tools_4, width=10, textvariable=var_bottom_x).grid(row=2, column=1)
entry_bottom_y = tkinter.Entry(frame_tools_4, width=10, textvariable=var_bottom_y).grid(row=2, column=3)

# crop button
button_crop = tkinter.Button(frame_tools_4, text="crop",
                             width=10, height=2,
                             fg='#4F051E', bg="#5AEE2F",
                             relief='raised', bd=2,
                             command=lambda: crop(var_top_x.get(), var_top_y.get(),
                                                  var_bottom_x.get(), var_bottom_y.get()))
button_crop.grid(row=1, column=4, rowspan=2)

# info panel
# info menu label
label_info = MenuLabel(frame_info, "Info Menu")
label_info.grid(row=0, column=0, columnspan=2)

# Numpy Histogram button
button_numpy_histogram = tkinter.Button(frame_info, text="Numpy Histogram",
                                        font='courier 12', height=1, width=18,
                                        relief=tkinter.RIDGE, bd=3,
                                        bg='#ad458f', fg='#ffaaff', padx=0,
                                        command=lambda: numpy_histogram()).grid(row=1, column=0)

# RGB histogram button
button_numpy_RGB = tkinter.Button(frame_info, text="RGB Histogram",
                                  font='courier 12', height=1, width=18,
                                  relief=tkinter.RIDGE, bd=3,
                                  bg='#ad458f', fg='#ffaaff', padx=0,
                                  command=lambda: rgb_histogram()).grid(row=1, column=1)


# save button
button_save = tkinter.Button(frame_save, text='Save',
                             bg='#26E48F', fg='#457603',
                             bd=3, relief="raised",
                             width=15, font='fixedsys 15',
                             command=lambda: save(output_name=var_save.get()))
# button_save.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

# entry to add save name and path
var_save = tkinter.StringVar()
entry_save = tkinter.Entry(frame_save, width=38, bg="#457898", textvariable=var_save)

button_save.grid(row=0, column=1)
entry_save.grid(row=0, column=0)

window.mainloop()