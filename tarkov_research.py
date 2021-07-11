import argparse
import logging
import threading
from collections import namedtuple
from time import sleep
from typing import List

import numpy as np
import cv2 as cv
from mss import mss
import pyautogui


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-wheel-click', action='store_true')
args = parser.parse_args()


logging.getLogger().setLevel(logging.INFO)

Point = namedtuple('Point', ['x', 'y'])


mon = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}


inv_p1 = Point(9, 262)
inv_p2 = Point(639, 954)
r_side = 62


GREY_RANGE = ([64, 64, 64, 0], [86, 86, 86, 255])
DARK_GREY_RANGE = ([41, 41, 41, 0], [66, 66, 66, 255])
OBJECT_RANGE = ([14, 14, 14, 0], [30, 30, 30, 255])

GREY_THRESHOLD = 300
DARK_GREY_THRESHOLD = 300
OBJECT_THRESHOLD = 100


def get_img_mask(img, lower_bound, upper_bound):
    lower_grey = np.array(lower_bound)
    upper_grey = np.array(upper_bound)

    return cv.inRange(img, lower_grey, upper_grey)


def init_inventory_matrix(upper_left_corner: Point, r_side: int):
    x1, y1 = upper_left_corner
    inv_matrix = np.ndarray(shape=(11, 10, 2, 2), dtype=int)
    for i in range(len(inv_matrix)):
        for j in range(len(inv_matrix[i])):
            inv_matrix[i][j] = [
                [x1 + r_side * j + j, y1 + r_side * i + i],
                [x1 + r_side * (j + 1) + j, y1 + r_side * (i + 1) + i]
            ]
    return inv_matrix


def draw_rectangles(image, inv_matrix, inv_mask, color, thickness):
    for i in range(len(inv_mask)):
        for j in range(len(inv_mask[i])):
            if inv_mask[i][j]:
                box = inv_matrix[i][j]
                cv.rectangle(image, *box, color, thickness)


def get_inv_mask(img_mask, inv_matrix, threshold):
    inv_mask = np.zeros(shape=(len(inv_matrix), len(inv_matrix[0])), dtype=bool)
    for i in range(len(inv_matrix)):
        for j in range(len(inv_matrix[i])):
            (x1, y1), (x2, y2) = inv_matrix[i][j]
            nonzeros = np.count_nonzero(img_mask[y1:y2, x1:x2])
            if nonzeros > threshold:
                inv_mask[i][j] = True
    return inv_mask


def get_box_center(box_x1, box_y1):
    return Point(box_x1 + r_side / 2, box_y1 + r_side / 2)


def delete_clicked_items_from_mask(inv_mask, clicked_indexes):
    for i, j in clicked_indexes:
        inv_mask[i][j] = False


def mouse_clicker(coords_queue: List[Point]):
    logging.info('mouse_clicker started')
    try:
        while True:
            with new_mouse_coords_added:
                logging.info('mouse_clicker waiting for coords')
                new_mouse_coords_added.wait()
                if stop_event.is_set():
                    logging.info('mouse_clicker stopping')
                    return
                while len(coords_queue):
                    coords = coords_queue.pop(0)
                    if args.debug or args.no_wheel_click:
                        pyautogui.click(coords.x, coords.y)
                    else:
                        pyautogui.click(coords.x, coords.y, button=pyautogui.MIDDLE)
                    logging.info(f'Mouse click: {pyautogui.position()}')
            pyautogui.moveTo(mon['width'], mon['height'] / 2)
    except pyautogui.FailSafeException as error:
        logging.info(error)


def get_item_inv_mask(img, inv_matrix):
    img_mask_hatching_grey = get_img_mask(img, *GREY_RANGE)
    img_mask_hatching_dark_grey = get_img_mask(img, *DARK_GREY_RANGE)
    img_mask_object = get_img_mask(img, *OBJECT_RANGE)

    inventory_mask_grey = get_inv_mask(
        img_mask_hatching_grey, inv_matrix, GREY_THRESHOLD
    )
    inventory_mask_dark_grey = get_inv_mask(
        img_mask_hatching_dark_grey, inv_matrix, DARK_GREY_THRESHOLD
    )
    inventory_mask_object = get_inv_mask(
        img_mask_object, inv_matrix, OBJECT_THRESHOLD
    )

    inventory_mask_background = np.bitwise_and(
        inventory_mask_grey, inventory_mask_dark_grey
    )
    inventory_mask_full_item = np.bitwise_and(
        inventory_mask_background, inventory_mask_object
    )

    if args.debug:
        cv.imshow('mask_hatching', img_mask_hatching_grey[inv_p1.y:inv_p2.y, inv_p1.x:inv_p2.x])
        cv.imshow('img_mask_hatching_dark_grey', img_mask_hatching_dark_grey[inv_p1.y:inv_p2.y, inv_p1.x:inv_p2.x])
        cv.imshow('img_mask_object', img_mask_object[inv_p1.y:inv_p2.y, inv_p1.x:inv_p2.x])

    return inventory_mask_full_item


def take_screenshots_until_found(inv_matrix):
    with mss() as sct:
        logging.info('Taking screenshots...')
        while True:
            screenshot = sct.grab(mon)
            img = np.array(screenshot)

            inv_mask_full_item = get_item_inv_mask(img, inv_matrix)

            if np.count_nonzero(inv_mask_full_item) > 0:
                logging.info('Not researched items found')
                return img, inv_mask_full_item


stop_event = threading.Event()
new_mouse_coords_added = threading.Condition()
mouse_coords_queue = []

mouse_click_thread = threading.Thread(target=mouse_clicker, args=[mouse_coords_queue])

inventory_matrix = init_inventory_matrix(inv_p1, r_side)
clicked_cells: List[tuple] = []

mouse_click_thread.start()
try:
    while True:
        scr_img, inventory_mask_full_item = take_screenshots_until_found(inventory_matrix)
        # Remove clicked item from the mask
        # (Ideally it should've been handled in np.where, but this seems more straightforward)
        delete_clicked_items_from_mask(inventory_mask_full_item, clicked_cells)

        if args.debug:
            draw_rectangles(scr_img, inventory_matrix, inventory_mask_full_item, (0, 0, 255), 2)
            cv.imshow('tarkov research', scr_img[inv_p1.y:inv_p2.y, inv_p1.x:inv_p2.x])
            cv.waitKey(0)

        item_indexes = np.where(inventory_mask_full_item)
        if len(item_indexes[0]) == 0:
            logging.info('No new items found, starting over')
            sleep(1)
            continue
        index_x, index_y = item_indexes[0][0], item_indexes[1][0]
        (x1, y1), _ = inventory_matrix[index_x][index_y]

        # One click per screenshot to skip large already researched items
        mouse_coords_queue.append(get_box_center(x1, y1))
        clicked_cells.append((index_x, index_y))

        with new_mouse_coords_added:
            new_mouse_coords_added.notify()
        sleep(1.1)
except KeyboardInterrupt:
    logging.info('KeyboardInterrupt, stopping execution')
except Exception as e:
    logging.error(e)
    raise e
finally:
    logging.info('Cleaning up...')
    stop_event.set()
    with new_mouse_coords_added:
        new_mouse_coords_added.notify()
    cv.destroyAllWindows()
    mouse_click_thread.join()
