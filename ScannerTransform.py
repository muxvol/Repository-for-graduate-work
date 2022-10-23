import sys
from typing import Tuple
import cupy as cp  # type: ignore
import yaml
from cupy._core import ndarray  # type: ignore


def load_camera_parameters(filename: str) -> Tuple[ndarray, ndarray]:
    cam_par = yaml.safe_load(open(filename)) 'Открытие файла с данными типа yaml (файл сканирования)'
    mtx = cp.array(             'Создание массива с использованием библиотеки cupy, оптимизирующей работу на графическом процессоре'
        [[cam_par['pinhole']['fx'],
          cam_par['pinhole']['skew'],
          cam_par['pinhole']['cx']],
         [0, cam_par['pinhole']['fy'], cam_par['pinhole']['cy']],
         [0, 0, 1]])
    distortion_k = cp.array([      'Создание массива с использованием библиотеки cupy, оптимизирующей работу на графическом процессоре'
        *cam_par['radial_tangential']['radial'],
        cam_par['radial_tangential']['t1'],
        cam_par['radial_tangential']['t2']])
    return mtx, distortion_k


def scanner_transform(lines: ndarray, calibrated_matrix: ndarray = None) -> Tuple[ndarray, ndarray]:
    if calibrated_matrix is None:
        calibrated_matrix = cp.eye(4) 'Матрица с единицами на ОБЕИХ диагоналях'
    """Убираем дисторсию в точках""" 'Дисторсия - искажение на снимке'
    x = lines[:, 0]
    y = lines[:, 1]
    ones = cp.ones_like(x) 'Массив х заполняется единицами'
    mtx, distortion_k = load_camera_parameters("intrinsics2.yaml")
    inv_m = cp.linalg.inv(mtx) 'Вычисление обратной матрицы'
    dy, dx, _ = inv_m @ cp.stack([y, x, ones], axis=0) 'Соединение матриц y, x, ones вдоль горизонтали или вдоль вертикали и умножение результата на обратную mtx?'
    k1, k2, p1, p2 = distortion_k
    r2 = dx**2 + dy**2
    r4 = r2**2
    ux = dx * (1 + k1 * r2 + k2 * r4) + p2 * (r2 + 2 * dx**2) + 2 * p1 * dx * dy
    uy = dy * (1 + k1 * r2 + k2 * r4) + 2 * p2 * dx * dy + p1 * (r2 + 2 * dy**2)
    fy, fx, _ = mtx @ cp.stack([uy, ux, ones])
    lines[:, 0] = fx
    lines[:, 1] = fy

    """Определяем параметры 3Д сканера и применяем их"""
    alpha = cp.radians(-20)
    beta = cp.radians(-35)
    y_speed = 700  # speed of platform mm/s
    b = -1500  # distance between camera and laser emitter

    line, cols = ux, uy 'ux = tg(phi), где phi - угол отклонения изображения луча от оптической оси в плоскости yОz'

    coef_n = b * cp.sin(alpha) 'коэффициент при тангенсе phi в числителе'
    bias_n = -b * cp.cos(alpha) 'свободный член в числителе'
    coef_d = cp.cos(alpha) + cp.sin(alpha) * cp.tan(beta)
    bias_d = cp.sin(alpha) - cp.cos(alpha) * cp.tan(beta)
    coef_y = -cp.tan(beta) 'коэффициент для преобразования z в y'
    coef_xz = cp.cos(alpha)
    coef_xy = -cp.sin(alpha)

    numerator = coef_n * line + bias_n
    denominator = coef_d * line + bias_d
    numerator[denominator == 0] = 0
    denominator[denominator == 0] = sys.float_info.epsilon 'Машинное эпсилон?'
    z = numerator / denominator
    y = z * coef_y
    x = cols * (z * coef_xz + y * coef_xy - 1)
    y0 = lines[:, 2] * y_speed
    y = y + y0
    cld = cp.stack([x, y, z, cp.ones_like(x)], axis=-1).T
    norms = cp.zeros((3, cld.shape[1])) 'Массив нулей формы 3 х dim[cld]'
    norms[0] = cols * 4
    norms[1] = 4
    norms[2] = 4
    cld = calibrated_matrix @ cld
    norms = calibrated_matrix[:3, :3] @ norms
    cld = cld[:3] 'Третья строка в cld?'
    return cld, norms
# Пробую добавить изменения в текст программы и запушить их

