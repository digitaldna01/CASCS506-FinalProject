// document.getElementById('features').addEventListener('submit', function (event) {
//     event.preventDefault();
//     // Display a loading message or spinner here
//     alert('Generating your scatter plot...');

//     // Submit the form asynchronously using AJAX
//     var form = this;
//     var xhr = new XMLHttpRequest();
//     xhr.open(form.method, form.action, true);
//     xhr.onload = function() {
//         if (xhr.status === 200) {
//             // Update the page with the new plot
//             document.body.innerHTML = xhr.responseText;
//         } else {
//             alert('An error occurred while generating the plot.');
//         }
//     };
//     xhr.send(new FormData(form));
// });



// document.getElementById("query-form").addEventListener("submit", async function(event) {
//     event.preventDefault(); // Prevent default form submission

//     // Get elements
//     const queryType = document.getElementById("query-type").value;
//     const textQuery = document.getElementById("text-query").value.trim();
//     const imageQuery = document.getElementById("image-query").files[0];
//     const weight = parseFloat(document.getElementById("weight").value);
//     const embeddingType = document.getElementById("embedding-type").value;
//     const resultsSection = document.getElementById("results-section");
//     const resultsList = document.getElementById("results-list");

//     // Reset previous results
//     resultsSection.style.display = "none";
//     resultsList.innerHTML = "";

//     // Validation based on query type
//     if (queryType === "text" && !textQuery) {
//         alert("Please enter a text query.");
//         return;
//     }

//     if (queryType === "image" && !imageQuery) {
//         alert("Please upload an image.");
//         return;
//     }

//     if (queryType === "hybrid") {
//         if (!textQuery || !imageQuery) {
//             alert("Please provide both text query and an image for a hybrid query.");
//             return;
//         }
//         if (isNaN(weight) || weight < 0.0 || weight > 1.0) {
//             alert("Please enter a weight between 0.0 and 1.0 for the hybrid query.");
//             return;
//         }
//     }

//      // Prepare form data for the request
//     const formData = new FormData();
//     formData.append("query_type", queryType);
//     formData.append("text_query", textQuery);
//     if (imageQuery) {
//         formData.append("image_query", imageQuery);
//     }
//     if (queryType === "hybrid") {
//         formData.append("weight", weight);
//     }
//     formData.append("embedding_type", embeddingType);

//     // Send the request to the backend
//     try {
//         const response = await fetch("/image_search", {
//             method: "POST",
//             body: formData,
//         });

//         if (!response.ok) {
//             throw new Error("Failed to fetch results.");
//         }

//         const data = await response.json();

//         // Display results
//         if (data.results && data.results.length > 0) {
//             resultsSection.style.display = "block";
//             data.results.forEach((result) => {
//                 const listItem = document.createElement("li");
//                 listItem.innerHTML = `
//                     <strong>Image:</strong> 
//                     <img src="${result.image_url}" alt="Result Image" style="max-width: 100px;"> 
//                     <br><strong>Score:</strong> ${result.similarity_score}`;
//                 resultsList.appendChild(listItem);
//             });
//         } else {
//             resultsSection.style.display = "block";
//             resultsList.innerHTML = "<li>No results found.</li>";
//         }
//     } catch (error) {
//         console.error("Error fetching results:", error);
//         alert("An error occurred while fetching results.");
//     }
// });
document.getElementById('features').addEventListener('submit', function (event) {
    event.preventDefault();

    let feature1 = document.getElementById('feature1').value;
    let feature2 = document.getElementById('feature2').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    // Prepare form data for the request
    const formData = new FormData();
    formData.append("feature1", feature1);
    formData.append("feature2", feature2);

    // Send the request to the backend
    fetch('/plot', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (response.ok) {
            return response.blob(); // Get the response as a Blob (binary large object)
        } else {
            throw new Error('Failed to generate plot');
        }
    })
    .then(blob => {
        // Create a URL for the image Blob and display it in an <img> tag
        const imgURL = URL.createObjectURL(blob);
        resultsDiv.innerHTML = `<img src="${imgURL}" alt="Generated Plot">`;
    })
    .catch(error => {
        console.error(error);
        resultsDiv.innerHTML = 'An error occurred while generating the plot.';
    });
});