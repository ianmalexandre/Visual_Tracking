from utils import *


def main1(ap, args):
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]

    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) <3:
        tracker = cv2.Tracker_create(args["tracker"].upper())

    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create,
            "medianflow": cv2.TrackerMedianFlow_create

        }

        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()


    # otherwise, grab a reference to the video file
    path = '../cars/' + args.get("video")
    gt = np.loadtxt('../cars/gt' + args.get("video") + '.txt', dtype=float, delimiter=',')
    all_files = os.listdir(path)
    all_files = sorted(list(set([filename for filename in all_files])))
    if all_files[0] == '.DS_Store':
        all_files.remove('.DS_Store')

    initBB = tuple(gt[0])

    aux2 = 0
    totalJack = np.zeros((1))
    error = 0
    for f in all_files:
        frame = cv2.imread(path + '/' + f)

        if frame is None:
            break

        if not math.isnan(gt[aux2][0]):
            cv2.rectangle(frame, (int(gt[aux2][0]), int(gt[aux2][1])), 
                (int(gt[aux2][2]),  int(gt[aux2][3])),
                (0, 0, 255), 2)
        (H, W) = frame.shape[:2]
        frame = imutils.resize(frame, width=W)
        (H, W) = frame.shape[:2]
        if initBB is not None:
            distx = int(initBB[2] - initBB[0])
            disty = int(initBB[3] - initBB[1])
            tracker.init(frame, initBB)
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                w = x+distx
                h = y+disty
                cv2.rectangle(frame, (x, y), (w, h),
                    (0, 255, 0), 2)
                
                Testbox = [x, y, w, h]
                if not math.isnan(gt[aux2][0]):
                    Jack = jaccard(gt[aux2], Testbox)
                    totalJack = np.append(totalJack, Jack)
                    print("Jaccard index = ", Jack)
                    if Jack == 0.0:
                        error = error + 1
                        print("Updating initial bounding box!")
                        tracker.clear()
                        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                        initBB = tuple(gt[aux2])
            elif not math.isnan(gt[aux2][0]):
                tracker.clear()
                tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                error = error + 1
                initBB = tuple(gt[aux2])

            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("Jaccard Index","{:.3f}".format(Jack))
            ]


            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        aux2 = aux2 + 1

    # close all windows
    cv2.destroyAllWindows()
    evaluate(gt, totalJack, error, args["tracker"], args["video"])

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default="car1",
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="mosse",
        help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    main1(ap, args)