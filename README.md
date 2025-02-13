<img src="data/drill_ship.png" style="width:300px; float:left">
<h1 style="text-align: center; color:teal">Data Science and Machine Learning for Planet Earth </h1>

# 📝 Task description

Your goal is to create the best possible predictor of sediment classes based on wireline log data. You are given one well. One section of the well has labels (the name of the lithologies), whereas the bottom part of the well has no labels. Your ultimate goal is to train an algorithm based on the section of the well that has labels to predict lithologies in the bottom part of the same well (where you don't have labels).

The performance of your algorithm will be determined based on how well you predict these unseen data (see more details about this in the notebook and below). A more detailed description of your task is given in the notebook.


# 🐟 Some notes about the data

Below is information about each features in your dataframe. This information could be useful to understand your task.

`DEPTH_WMSF`: This represents the depth of measurement below the seafloor, expressed in meters. It allows you to correlate geological features with depth.

`HCGR`: Total gamma ray counts, measured in counts per second (cps), indicate the overall radioactivity of the formation, often used to identify different rock types.

`HFK`: Potassium counts reflect the contribution of potassium to total gamma radiation. It's used to identify clay content and alteration in rocks.

`HTHO`: Thorium counts measure thorium's contribution to total gamma radiation, aiding in differentiating between siliciclastic and carbonate rocks.

`HURA`: Uranium counts assess uranium's contribution to total gamma radiation. It's particularly useful in organic-rich environments and for identifying hydrocarbon presence.

`IDPH`: Deep Phasor Dual Induction–Spherically Focused Resistivity measures how well electricity penetrates rocks at greater depths, indicating porosity and fluid content in ohms.

`IMPH`: Medium Phasor Dual Induction–Spherically Focused Resistivity, also in ohms, assesses electrical penetration at intermediate depths, providing insights into rock texture and fluid saturation.

`SFLU`: Shallow Phasor Dual Induction–Spherically Focused Resistivity measures electrical resistivity at shallower depths, crucial for understanding near-surface rock properties and fluid content.

`munsell_color`: This property is a `string` that contains the exact color identification based on the <a href="https://munsell.com/about-munsell-color/how-color-notation-works/how-to-read-color-chart/">Munsell color chart</a>. It describes sediment colors, aiding in identifying mineral content and environmental conditions, and thus lithology. Munsell color charts are used by sedimentologists and many other scientists. Converting the color to other system is not trivial: most of the time the color name itself is used, though conversion tables do exist (<a href="https://github.com/samdroid-apps/python-munsell">see this for instance</a>)

`lithology`: this is your target value, i.e. what you are trying to predict. It is a string representing the name of the lithology. You don't need to understand the lithology name for this assessment, but if you want to, you can easily google it to find out what each name means.