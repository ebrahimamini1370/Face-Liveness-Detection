# Face-Liveness-Detection


<body>

  <h1>Introduction</h1>
   The goal of this project was to develop a Face Liveness Detection
   application using a Local Binary Pattern approach and,
   using the same approach, develop a real time Face Liveness Detection
   application. For the Face Liveness Detection task, I have used
   the algorithm, which is proposed by the article <a href="https://ieeexplore.ieee.org/abstract/document/8243913">Face Liveness
   Detection Based on Enhanced Local Binary Patterns</a>. Based on this
   algorithm, we must prepare a dataset of live and fake images in gray scale. Then we
   must extract the ELBP features from the images, and finally train a
   SVM as a binary classifier.

  <h1>Run</h1>
  To run this program, the following steps must be done.
  <ol>
  <li>Prepare database : </li>
    Dataset images must be put in the 'Dataset' folder. The reference article,
    has used the <a href="http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html">NUAA Dataset</a>.

  <li>main.py: </li>
    This file, performs two functions. the first function goes through all images
    in dataset, computes their enhace local binary patterns, and prepares a database
    appropriate to accomplish binary classification task, and finally saves
    the database in 'liveness_detection_database.pkl' file.
    Then, the second function, trains a support vector machine on database and
    saves the classifier in 'svm_clf.pkl' file.

  <li>RealTime.py: </li>
    Running this file, turns on the camera, and performs face liveness
    detection task on each frame.
  </ol>

  <h1>Other files in the project folder</h1>
  <ol>
  <li>utils.py: </li>
    This file includes all libraries and functions we need to run
    the project.

  <li>simhei.ttf: </li>
    This is a font file we use to show the results of liveness detection
    on the real-time frame. If you would like to show results in any
    language other than English, you can provide your appropriate font file,
    and copy its path_file in the 'RealTime.py' file.

  <li>haarcascade_frontalface_default.xml: </li>
    An appropriate face detection tool for real-time applications. However, it
    can be replaced by more exactly face detectors, deep learning-based ones for example.
  </ol>

  <h1>References</h1>
  <ul>
    <li> <a href="https://ieeexplore.ieee.org/abstract/document/8243913"> [1] X. Liu, R. Lu, W. Liu, "Face liveness detection based on enhanced local binary patterns",
       Chinese Automation Congress, pp. 6301-6305, 2017.  </a></li>
    <li> <a href="https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern"> [2] skiti-image-local_binary_pattern</a></li>
  </ul>

</body>
