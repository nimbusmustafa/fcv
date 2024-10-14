import cv2
cap = cv2.VideoCapture(0)
# Define the codec using FourCC code
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create a VideoWriter object
output_file = 'output_video.avi'  # The name of the output file

video_writer = cv2.VideoWriter(output_file, fourcc,30,(640,480))

# Open the video source (0 for webcam, or provide a file path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Write the frame to the video file
    video_writer.write(frame)

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
video_writer.release()
cv2.destroyAllWindows()
