from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt


def apply_hysteresis_threshold(image, low, high):
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/filters/thresholding.py
    # 上記のscikit imageからそのまま使用
    """Apply hysteresis thresholding to ``image``.

    This algorithm finds regions where ``image`` is greater than ``high``
    OR ``image`` is greater than ``low`` *and* that region is connected to
    a region greater than ``high``.

    Parameters
    ----------
    image : (M[, ...]) ndarray
        Grayscale input image.
    low : float, or array of same shape as ``image``
        Lower threshold.
    high : float, or array of same shape as ``image``
        Higher threshold.

    Returns
    -------
    thresholded : (M[, ...]) array of bool
        Array in which ``True`` indicates the locations where ``image``
        was above the hysteresis threshold.

    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])

    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           :DOI:`10.1109/TPAMI.1986.4767851`
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]

    # fill holes
    thresholded = ndi.binary_fill_holes(thresholded)
    thresholded = thresholded.astype('float32') # bool -> float32
    
    return thresholded


from skimage.measure import label, regionprops
def remove_small_islands(image, volume_threshold):  
    # 画像をラベル付け
    labeled_image, num_objects = label(image, return_num=True, connectivity=3) # connectivity=3は26近傍
    print("Number of objects:", num_objects)

    # Calculate the volume of each object
    volumes = ndi.sum(image, labeled_image, range(1, num_objects+1))
    print(volumes)

    # Plot histogram of volumes
    plt.hist(volumes, bins=100, color='c')
    plt.title('Volume Distribution of Objects')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')
    plt.show()


    # 各ラベル付けされたオブジェクトのプロパティを計算
    object_properties = regionprops(labeled_image)

    # Iterate over each object and print its properties
    for obj in object_properties:
        print("Object Label:", obj.label)
        print("Object Volume:", obj.area)
        print("Object Diameter:", obj.equivalent_diameter)
        print("Object Bounding Box:", obj.bbox)
        print("Object Centroid:", obj.centroid)
#         print("Object axis_minor_length:", obj.axis_minor_length)
#         print("Object axis_major_length:", obj.axis_major_length)
        print("------------------------------")

        # ラベル付けされたオブジェクトを可視化
        img = np.where(labeled_image == obj.label, 1, 0)
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.imshow(np.max(img, axis=2), alpha=0.5)
        plt.subplot(132)
        plt.imshow(np.max(img, axis=1), alpha=0.5)
        plt.subplot(133)
        plt.imshow(np.max(img, axis=0), alpha=0.5)
        plt.title("Object Label: " + str(obj.label))
        plt.show()

    # object volumeが閾値ボクセル未満のobjectは除去
    new_image = np.copy(image)
    for obj in object_properties:
        if obj.area < volume_threshold:
            new_image[labeled_image == obj.label] = 0

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(np.max(new_image, axis=2), cmap="gray")
    plt.imshow(np.max(image, axis=2), alpha=0.5)
    plt.subplot(132)
    plt.imshow(np.max(new_image, axis=1), cmap="gray")
    plt.imshow(np.max(image, axis=1), alpha=0.5)
    plt.subplot(133)
    plt.imshow(np.max(new_image, axis=0), cmap="gray")
    plt.imshow(np.max(image, axis=0), alpha=0.5)
    plt.suptitle("Only the main cerebral arteries are extracted")
    plt.show()

    return new_image.astype('float32') # bool型からfloat32型に変換


def match_origin_and_direction(original_img, target_img):
    print("original image header", original_img)
    print("target image header", target_img)
    
    # Get the origin and direction of the original image
    original_origin = original_img.origin
    original_direction = original_img.direction
    
    # Get the origin and direction of the target image
    target_origin = target_img.origin
    target_direction = target_img.direction
    
    # Check if origin and direction are the same
    if np.allclose(original_origin, target_origin) and np.allclose(original_direction, target_direction):
        print("The origin and direction are already matched.")
        return target_img
    else:
        print("The origin and/or direction are not matched. Updating to match the original image.")
        # Set the origin and direction of the target image to match the original image
        target_img.set_origin(original_origin)
        target_img.set_direction(original_direction)
        return target_img

