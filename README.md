# OCV_KochNN
<h1 align="center">Hy 👋, ich bin Christian Wiegel</h1>

<p align="left">
Dieses Programm berechnet die Koch-Kurve, so dass jedem x-Wert eindeutig ein y-Wert zugeordnet wird mit Hilfe zwei neuronaler Netze in OpenCV.</p>

<p align="left">
Um das rekursive Programmieren zu erlernen, setzen sich Informatik-Studenten mit der Koch-Kurve auseinander. Warum eine Funktion die noch nicht einmal stueckweise glatt ist als "Kurve" bezeichnet wird, wird mir immer ein Rätsel bleiben...</p>

<p align="left">
Das Programm sucht zunaechst Punkte der Kurve in der aufeinander folgenden Reihenfolge. Jeder Punkt hat einen x- und einen y-Wert und ausserdem ist die zurückgelegte Strecke am aktuellen Punkt bekannt.
</p>

<p align="left">
Das Programm besteht aus zwei neuronalen Netzen, die in diesem Text mit den Funktionen <i>f</i> und <i>g</i> beschrieben werden. Die eine Funktion / das erste neuronale Netz <i>s = f(x)</i> ordnet jedem x-Wert eine zurueckgelegte Strecke 's' zu; die andere Funktion / das zweite neuronale Netz <i>y = g(s)</i> ordnet jeder zurueckgelegten Strecke einen y-Wert zu.
</p>

<p align="left">
Da sich die Koch-Kurve aus linearen Werten zusammensetzt wurde das Leaky-ReLU-Neuronen-Modell gegenueber Neuronen-Modellen, die mit Sigmoid-Funktionen arbeiten, favorisiert.
</p>

<p align="left">
Die geplottete Funktion sieht folgendermassen aus:
</p>

<p align="left">
<img src="https://github.com/robovisioneer/OCV_KochNN/blob/main/Kochkurve.png" alt="kochkurve" width="684" height="398"/>
</p>

<p align="left">
Eine Koch-Kurve ist klar zu erkennen... Mit dem Optimieren der Topologie und den Parametern des Lernalgorithmus laesst sich das Ergebnis sicher noch verbessern; ich bin mit dem Ergebnis aber sehr zufrieden.
</p>


<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.w3schools.com/cpp/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> </p>

