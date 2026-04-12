const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const previewImage = document.getElementById("previewImage");
const predictButton = document.getElementById("predictButton");
const resultText = document.getElementById("result");

let selectedFile = null;
const API_ENDPOINT = "/predict";

function updateResult(message, isError = false) {
    resultText.textContent = message;
    resultText.classList.toggle("error", isError);
}

function updatePreview(file) {
    if (!file) {
        previewImage.removeAttribute("src");
        previewImage.classList.remove("visible");
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewImage.classList.add("visible");
    };
    reader.readAsDataURL(file);
}

function setLoadingState(isLoading) {
    predictButton.disabled = isLoading || !selectedFile;
    if (isLoading) {
        updateResult("Generating caption... ⏳");
    }
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) {
        selectedFile = null;
        fileName.textContent = "No image selected";
        updatePreview(null);
        predictButton.disabled = true;
        updateResult("Upload an image and click Generate Caption to begin.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    updatePreview(file);
    predictButton.disabled = false;
    updateResult("Ready to generate a caption.");
});

predictButton.addEventListener("click", async () => {
    if (!selectedFile) {
        updateResult("Please select an image before generating a caption.", true);
        return;
    }

    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append("image", selectedFile);

        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => null);
            throw new Error(errorBody?.message || "Unable to generate caption.");
        }

        const data = await response.json();
        updateResult(data.prediction || "No caption returned from the server.");
    } catch (error) {
        console.error("Caption generation failed:", error);
        updateResult("An error occurred while generating the caption. Please try again.", true);
    } finally {
        setLoadingState(false);
    }
});

// Initialize button state
predictButton.disabled = true;