/**
 * Frame Animation for Clipper
 * This script handles frame-by-frame animation for video previews
 *
 * Note: This animation system uses pre-rendered frames that already have the
 * crop regions correctly positioned. The frames are captured directly from
 * the main display, ensuring that the crop regions are exactly the same as
 * what the user sees when manually navigating through frames.
 *
 * For clip previews, only frames within the clip range (start_frame to end_frame)
 * are included in the animation, respecting the clip boundaries.
 *
 * The crop regions are interpolated between keyframes in Python before the frames
 * are sent to JavaScript. This ensures that the crop regions move smoothly between
 * keyframes during animation playback, just as they do when navigating frames manually.
 */

class FrameAnimator {
  constructor(containerId, frames, startFrame = 0, endFrame = null) {
    this.containerId = containerId;
    this.frames = frames;
    this.currentFrameIndex = 0;
    this.isPlaying = false;
    this.animationSpeed = 1.0;
    this.frameInterval = null;
    this.startFrame = startFrame;
    this.endFrame = endFrame || frames.length - 1;

    // Preload all images to ensure smooth playback
    this.preloadedImages = [];
    this.preloadingComplete = false;

    console.log(
      `Creating animator for ${containerId} with ${frames.length} frames`
    );

    // Initialize the container
    this.initialize();
  }

  initialize() {
    console.log(`Initializing animation for container: ${this.containerId}`);

    // Find the container - try multiple methods to ensure we find it
    let container = document.getElementById(this.containerId);

    if (!container) {
      console.error(
        `Container with ID ${this.containerId} not found on first attempt`
      );

      // Try to find it by querying for div elements with the ID
      const divs = document.querySelectorAll(`div[id="${this.containerId}"]`);
      if (divs.length > 0) {
        container = divs[0];
        console.log(
          `Found container by querying divs with ID ${this.containerId}`
        );
      }
    }

    if (!container) {
      console.error(
        `Container with ID ${this.containerId} still not found, animation cannot initialize`
      );
      // Log all div IDs for debugging
      const allDivs = document.querySelectorAll("div[id]");
      console.log(
        "Available div IDs:",
        Array.from(allDivs).map((div) => div.id)
      );
      return;
    }

    // Clear the container
    container.innerHTML = "";

    // Create a loading message
    const loadingMsg = document.createElement("div");
    loadingMsg.id = `${this.containerId}-loading`;
    loadingMsg.textContent = `Preloading ${this.frames.length} frames starting from frame ${this.startFrame}...`;
    loadingMsg.style.padding = "10px";
    loadingMsg.style.marginBottom = "10px";
    loadingMsg.style.backgroundColor = "#f0f0f0";
    loadingMsg.style.borderRadius = "4px";
    container.appendChild(loadingMsg);

    // Create image element
    const img = document.createElement("img");
    img.id = `${this.containerId}-frame`;
    img.style.width = "100%";
    img.style.display = "none"; // Hide until preloading is complete
    container.appendChild(img);

    // Create controls
    const controls = document.createElement("div");
    controls.style.marginTop = "10px";
    controls.style.display = "flex";
    controls.style.alignItems = "center";

    // Play button
    const playButton = document.createElement("button");
    playButton.id = `${this.containerId}-play`;
    playButton.textContent = "Play";
    playButton.style.marginRight = "10px";
    playButton.style.padding = "5px 10px";
    playButton.style.backgroundColor = "#4CAF50";
    playButton.style.color = "white";
    playButton.style.border = "none";
    playButton.style.borderRadius = "4px";
    playButton.style.cursor = "pointer";
    playButton.style.display = "none"; // Hide until preloading is complete

    // Frame counter
    const frameCounter = document.createElement("span");
    frameCounter.id = `${this.containerId}-counter`;
    frameCounter.textContent = `Frame: ${this.startFrame} / ${this.endFrame}`;

    // Add elements to controls
    controls.appendChild(playButton);
    controls.appendChild(frameCounter);
    container.appendChild(controls);

    // Add event listeners
    playButton.addEventListener("click", () => this.togglePlayback());

    // Preload all images for smooth playback
    this.preloadImages()
      .then(() => {
        // Update UI after preloading
        const loadingElement = document.getElementById(
          `${this.containerId}-loading`
        );
        if (loadingElement) {
          loadingElement.textContent = `Preloading complete! Ready to play ${this.frames.length} frames starting from frame ${this.startFrame}.`;
          loadingElement.style.backgroundColor = "#d4edda";
        }

        // Show the image and play button
        const imgElement = document.getElementById(`${this.containerId}-frame`);
        if (imgElement) {
          imgElement.style.display = "block";
        }

        const playButtonElement = document.getElementById(
          `${this.containerId}-play`
        );
        if (playButtonElement) {
          playButtonElement.style.display = "block";
        }

        // Display first frame
        this.updateFrame();

        console.log(`Animation initialized for ${this.containerId}`);
      })
      .catch((error) => {
        console.error(`Error preloading images: ${error}`);

        // Show error message
        const loadingElement = document.getElementById(
          `${this.containerId}-loading`
        );
        if (loadingElement) {
          loadingElement.textContent =
            "Error preloading images. Please try again.";
          loadingElement.style.backgroundColor = "#f8d7da";
        }
      });
  }

  // Preload all images for smooth playback
  async preloadImages() {
    console.log(
      `Preloading ${this.frames.length} frames for ${this.containerId}`
    );

    // Create a progress indicator
    const loadingElement = document.getElementById(
      `${this.containerId}-loading`
    );

    // Create all image objects and store them
    const totalFrames = this.frames.length;
    const batchSize = 10; // Process images in batches to avoid overwhelming the browser

    for (let i = 0; i < totalFrames; i += batchSize) {
      // Update progress message
      if (loadingElement) {
        const progress = Math.min(100, Math.round((i / totalFrames) * 100));
        loadingElement.textContent = `Preloading frames: ${progress}% complete...`;
      }

      // Process a batch of images
      const batch = [];
      for (let j = 0; j < batchSize && i + j < totalFrames; j++) {
        const index = i + j;
        batch.push(this.preloadImage(index));
      }

      // Wait for the current batch to complete
      await Promise.all(batch);
    }

    this.preloadingComplete = true;
    console.log(
      `Preloaded ${this.preloadedImages.length} frames for ${this.containerId}`
    );
  }

  // Preload a single image
  preloadImage(index) {
    return new Promise((resolve) => {
      const img = new Image();

      // Set up event handlers
      img.onload = () => {
        this.preloadedImages[index] = img;
        resolve();
      };

      img.onerror = () => {
        console.error(`Failed to load image at index ${index}`);
        resolve(); // Resolve anyway to continue the process
      };

      // Set a timeout to resolve even if the image doesn't load
      const timeout = setTimeout(() => {
        if (!this.preloadedImages[index]) {
          console.warn(`Image load timed out for index ${index}`);
          resolve();
        }
      }, 1000);

      // Start loading the image
      img.src = `data:image/jpeg;base64,${this.frames[index]}`;
    });
  }

  updateFrame() {
    const img = document.getElementById(`${this.containerId}-frame`);
    if (!img) {
      console.error(`Image element not found for ${this.containerId}`);
      return;
    }

    // Use the preloaded image if available
    if (this.preloadedImages[this.currentFrameIndex]) {
      img.src = this.preloadedImages[this.currentFrameIndex].src;
    } else {
      // Fallback to base64 data
      img.src = `data:image/jpeg;base64,${this.frames[this.currentFrameIndex]}`;
    }

    // Update frame counter - calculate the actual frame number based on startFrame
    const frameCounter = document.getElementById(`${this.containerId}-counter`);
    if (frameCounter) {
      // Calculate the actual frame number by adding the startFrame to the current index
      const actualFrameNumber = this.startFrame + this.currentFrameIndex;

      // Show both the actual frame number and the animation progress
      frameCounter.textContent = `Frame: ${actualFrameNumber} (${
        this.currentFrameIndex + 1
      }/${this.frames.length})`;
    }
  }

  togglePlayback() {
    if (this.isPlaying) {
      this.stopAnimation();
    } else {
      this.startAnimation();
    }
  }

  startAnimation() {
    if (this.isPlaying) return;

    this.isPlaying = true;
    const frameDelay = 1000 / (24 * this.animationSpeed); // 24 fps * speed

    // Update play button
    const playButton = document.getElementById(`${this.containerId}-play`);
    if (playButton) {
      playButton.textContent = "Pause";
    }

    // Start animation loop
    this.frameInterval = setInterval(() => {
      // Advance to next frame
      this.currentFrameIndex =
        (this.currentFrameIndex + 1) % this.frames.length;
      this.updateFrame();
    }, frameDelay);

    console.log(`Animation started for ${this.containerId}`);
  }

  stopAnimation() {
    if (!this.isPlaying) return;

    this.isPlaying = false;
    clearInterval(this.frameInterval);

    // Update play button
    const playButton = document.getElementById(`${this.containerId}-play`);
    if (playButton) {
      playButton.textContent = "Play";
    }

    console.log(`Animation stopped for ${this.containerId}`);
  }

  setSpeed(speed) {
    this.animationSpeed = speed;

    // If currently playing, restart with new speed
    if (this.isPlaying) {
      this.stopAnimation();
      this.startAnimation();
    }
  }
}

// Global variable to store animator instances
window.animators = {};

// Function to initialize animation when called from Streamlit
function initializeAnimation(
  containerId,
  framesData,
  startFrame = 0,
  endFrame = null
) {
  console.log(
    `Initializing animation for ${containerId} with ${framesData.length} frames, startFrame=${startFrame}, endFrame=${endFrame}`
  );

  try {
    // Check if we already have an animator for this container
    if (window.animators[containerId]) {
      console.log(`Animator for ${containerId} already exists, destroying it`);
      // Clean up any existing animation
      if (window.animators[containerId].isPlaying) {
        window.animators[containerId].stopAnimation();
      }
      // Remove reference to allow garbage collection
      delete window.animators[containerId];
    }

    // Validate inputs
    if (!framesData || framesData.length === 0) {
      console.error("No frames provided for animation");
      return null;
    }

    if (startFrame === undefined || startFrame === null) {
      console.warn("No start frame provided, defaulting to 0");
      startFrame = 0;
    }

    if (endFrame === undefined || endFrame === null) {
      console.warn(
        `No end frame provided, defaulting to startFrame + framesData.length - 1 (${
          startFrame + framesData.length - 1
        })`
      );
      endFrame = startFrame + framesData.length - 1;
    }

    // Create a new animator and store it in the global variable
    window.animators[containerId] = new FrameAnimator(
      containerId,
      framesData,
      startFrame,
      endFrame
    );

    console.log(`Successfully created animator for ${containerId}`);
    return window.animators[containerId];
  } catch (e) {
    console.error(`Error initializing animation: ${e.message}`);
    console.error(e.stack);

    // Try to display error in the container
    try {
      const container = document.getElementById(containerId);
      if (container) {
        container.innerHTML = `<div style="color: red; padding: 20px;">
          <h3>Animation Error</h3>
          <p>${e.message}</p>
          <p>Please try refreshing the page or contact support.</p>
        </div>`;
      }
    } catch (displayError) {
      console.error("Error displaying error message:", displayError);
    }

    return null;
  }
}

// Function to start animation directly
function startAnimation(containerId) {
  if (window.animators && window.animators[containerId]) {
    window.animators[containerId].startAnimation();
  } else {
    console.error(`No animator found for ${containerId}`);
  }
}

// Function to stop animation directly
function stopAnimation(containerId) {
  if (window.animators && window.animators[containerId]) {
    window.animators[containerId].stopAnimation();
  } else {
    console.error(`No animator found for ${containerId}`);
  }
}

// Make sure the script is loaded
console.log("Frame animation script loaded");

// Add a global function to check if containers exist
window.checkAnimationContainers = function () {
  console.log("Checking for animation containers...");
  const containers = document.querySelectorAll("div[id]");
  console.log(
    "All div elements with IDs:",
    Array.from(containers).map((div) => div.id)
  );

  // Specifically check for our animation containers
  const animContainer = document.getElementById("animation-container");
  console.log("animation-container exists:", !!animContainer);

  const previewContainer = document.getElementById(
    "preview-animation-container"
  );
  console.log("preview-animation-container exists:", !!previewContainer);

  return {
    found: {
      "animation-container": !!animContainer,
      "preview-animation-container": !!previewContainer,
    },
  };
};
