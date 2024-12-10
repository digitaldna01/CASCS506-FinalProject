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

document.getElementById('add').addEventListener('submit', function (event) {
    event.preventDefault();

    let resultsDiv = document.getElementById('add_info');
    resultsDiv.innerHTML = '';

    //const formData = new FormData();

    // const form = document.getElementById('add');
    // const formData = new FormData(form);

    const formData = {
        "radius_mean": parseFloat(document.getElementById('rad_mean').value),
        "texture_mean": parseFloat(document.getElementById('text_mean').value),
        "perimeter_mean": parseFloat(document.getElementById('per_mean').value),
        "area_mean": parseFloat(document.getElementById('area_mean').value),
        "smoothness_mean": parseFloat(document.getElementById('smooth_mean').value),
        "compactness_mean": parseFloat(document.getElementById('comp_mean').value),
        "concavity_mean": parseFloat(document.getElementById('conc_mean').value),
        "symmetry_mean": parseFloat(document.getElementById('sym_mean').value),
        "fractal_dimension_mean": parseFloat(document.getElementById('frac_dim_mean').value),

        "radius_se": parseFloat(document.getElementById('rad_se').value),
        "texture_se": parseFloat(document.getElementById('text_se').value),
        "perimeter_se": parseFloat(document.getElementById('per_se').value),
        "area_se": parseFloat(document.getElementById('area_se').value),
        "smoothness_se": parseFloat(document.getElementById('smooth_se').value),
        "compactness_se": parseFloat(document.getElementById('comp_se').value),
        "concavity_se": parseFloat(document.getElementById('conc_se').value),
        "symmetry_se": parseFloat(document.getElementById('sym_se').value),
        "fractal_dimension_se": parseFloat(document.getElementById('frac_dim_se').value),

        "radius_worst": parseFloat(document.getElementById('rad_worst').value),
        "texture_worst": parseFloat(document.getElementById('text_worst').value),
        "perimeter_worst": parseFloat(document.getElementById('per_worst').value),
        "area_worst": parseFloat(document.getElementById('area_worst').value),
        "smoothness_worst": parseFloat(document.getElementById('smooth_worst').value),
        "compactness_worst": parseFloat(document.getElementById('comp_worst').value),
        "concavity_worst": parseFloat(document.getElementById('conc_worst').value),
        "symmetry_worst": parseFloat(document.getElementById('sym_worst').value),
        "fractal_dimension_worst": parseFloat(document.getElementById('frac_dim_worst').value)
    };

    console.log(formData);

    fetch('/user_info', {
        method: 'POST',
        //body: formData,
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
        .then(response => {
            if (response.ok) {
                // return response.json(); // Parse the JSON response
                return response.blob();
            } else {
                throw new Error('Failed to add data');
            }
        })
        .then(blob => {
            // Create a URL for the image Blob and display it in an <img> tag
            const imgURL = URL.createObjectURL(blob);
            resultsDiv.innerHTML = `<img src="${imgURL}" alt="Generated Plot">`;
        })
        .catch(error => {
            console.error(error);
            alert('An error occurred while adding the data.');
        });
});