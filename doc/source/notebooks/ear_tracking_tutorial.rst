Ear tracking tutorial
=====================

.. code:: ipython2

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    import alinea.ear_tracking.ear_tracking_library_standalone  as et_lib
    import alinea.ear_tracking.standalone_library as st_lib
    

Get example images and parameters needed for binarisation and ear
tracking

.. code:: ipython2

    output_folder = os.path.join(os.path.expanduser('~'), 'ear_tracking_results')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    # Read images' information and create output folders
    # TODO read images from zenodo
    images_folder = os.path.join(os.path.expanduser('~'), 'Bureau',
                                 'ear_tracking_input')
    
    img_desc, parameters = st_lib.init(images_folder, output_folder)
    

Selection of one plant and one imaging task

.. code:: ipython2

    plant = img_desc.keys()[0]
    task = img_desc[plant].keys()[0]
    
    results_folder = os.path.join(output_folder, str(plant), 'results')
    cabin = img_desc[plant][task]['cabin']
    task_folder = os.path.join(output_folder, str(plant), str(task))

Read images and show them

.. code:: ipython2

    images = st_lib.read_images(img_desc, plant, task)
    fig, axes = plt.subplots(nrows=5, ncols=3)
    axlist = fig.get_axes()
    axlist[1].imshow(images["top"][0])
    for i, side_img in enumerate(images["side"].values()):
        axlist[i+3].imshow(side_img)
    
    plt.show()



.. image:: ear_tracking_tutorial_files%5Cear_tracking_tutorial_6_0.png



.. image:: ear_tracking_tutorial_files%5Cear_tracking_tutorial_6_1.png


Calcul binaries and show them

.. code:: ipython2

    binaries, mask_top_center = st_lib.binaries_calculation(images, cabin, parameters)
    
    fig, axes = plt.subplots(nrows=5, ncols=3)
    axlist = fig.get_axes()
    axlist[1].imshow(binaries["top"][0])
    for i, side_img in enumerate(binaries["side"].values()):
        axlist[i+3].imshow(side_img)
    
    plt.show()



.. image:: ear_tracking_tutorial_files%5Cear_tracking_tutorial_8_0.png


.. code:: ipython2

    existing_angles = binaries["side"].keys()
    angles_to_keep, result_img, top_log = et_lib.top_analyse(binaries["top"][0], existing_angles, mask_top_center)
    plt.imshow(result_img)
    plt.show()



.. image:: ear_tracking_tutorial_files%5Cear_tracking_tutorial_9_0.png


.. code:: ipython2

    kept_positions = np.empty([0, 3], 'int')
    useful_kept_images = np.empty([0], 'int')
    for angle in angles_to_keep:
        positions, imagesUtiles, debug = et_lib.side_analyse(binaries["side"][angle],
                                                             images["side"][angle],
                                                             angle,
                                                             task_folder,
                                                             parameters[cabin]["side"]["pot_height"],
                                                             parameters[cabin]["side"]["pot_width"],)
        kept_positions = np.append(kept_positions, positions, axis=0)
        useful_kept_images = np.append(useful_kept_images, imagesUtiles, axis=0)

.. code:: ipython2

    values = np.empty([0, 2], 'int')
    im = np.empty([0], 'int')
    for i in range(len(kept_positions)):
        for j in range(kept_positions[i, 2]):
            values = np.append(values, [kept_positions[i, 0:2]], axis=0)
            im = np.append(im, useful_kept_images[i])
    mean_pos, finales_positions, final_kept_images = et_lib.robust_mean(values, im)
    finales_positions = np.append(finales_positions, np.array([final_kept_images]).T, 1)

.. code:: ipython2

    imageFinale = images["side"][finales_positions[0, 2]].copy()
    if not (mean_pos == [-1, -1]).all():
        imageFinale[mean_pos[0] - 10:mean_pos[0] + 11, mean_pos[1] - 10:mean_pos[1] + 11, :] = [0, 255, 255]
    elif finales_positions.shape[0] == 2:
        for pos in finales_positions:
            imageFinale[pos[0] - 10:pos[0] + 11, pos[1] - 10:pos[1] + 11, :] = [0, 255, 255]
    plt.imshow(imageFinale)
    plt.show()



.. image:: ear_tracking_tutorial_files%5Cear_tracking_tutorial_12_0.png


