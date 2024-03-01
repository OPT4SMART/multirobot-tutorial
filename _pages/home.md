---
layout: splash_custom
permalink: /
title: <p style="font-size:42px;">A Tutorial on Distributed Optimization for <br> Cooperative Robotics</p>
header:
  overlay_color: "#5e616c"
  overlay_image: /images/home_banner.png
#   actions:
#     - label: "<i class='fas fa-download'></i> Install now"
#       url: "/docs/quick-start-guide/"
defaults:
  # _pages
  - scope:
      path: ""
      type: pages
    values:
excerpt:
  from Setups and Algorithms to Toolboxes and Research Directions
feature_row:
  - image_path: /images/disropt.jpg
    alt: "DISROPT"
    title: "DISROPT"
    excerpt: "Here, we will delve into DISROPT basics and the implementation of the 'Charging of Plug-in Electric Vehicles' scenario."
    # excerpt: "Everything from the menus, sidebars, comments, and more can be configured or set with YAML Front Matter."
    url: "/disropt"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /images/choirbot.png
    alt: "ChoiRbot"
    title: "ChoiRbot"
    excerpt: "In this tutorial, we will delve into ChoiRbot basics and the implementation of the 'Task Allocation' scenario using Turtlebot3 and Gazebo simulator."
    url: "/docs/layouts/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: /images/crazychoir.png
    alt: "CrazyChoir"
    title: "CrazyChoir"
    excerpt: "In this tutorial, we will delve into CrazyChoir basics and the implementation of the 'Target Surveillance' scenario using Crazyflie and Webots simulator."
    url: "/docs/license/"
    btn_class: "btn--primary"
    btn_label: "Learn more"      

---

{% include feature_row %}


# Abstract

Several interesting problems in multi-robot systems can be cast in the framework of distributed optimization. Examples include multi-robot task allocation, vehicle routing, target protection and surveillance. While the theoretical analysis of distributed optimization algorithms has received significant attention, its application to cooperative robotics has not been investigated in detail. In this paper, we show how notable scenarios in cooperative robotics can be addressed by suitable distributed optimization setups. Specifically, after a brief introduction on the widely investigated consensus optimization (most suited for data analytics) and on the partition-based setup (matching the graph structure in the optimization), we focus on two distributed settings modeling several scenarios in cooperative robotics, i.e., the so-called constraint-coupled and aggregative optimization frameworks. For each one, we consider use-case applications, and we discuss tailored distributed algorithms with their convergence properties. Then, we revise state-of-the-art toolboxes allowing for the implementation of distributed schemes on real networks of robots without central coordinators. For each use case, we discuss their implementation in these toolboxes and provide simulations and real experiments on networks of heterogeneous robots.

***

<!-- You can cite this work using the following

    @article{testa2023tutorial,
      title={A Tutorial on Distributed Optimization for Cooperative Robotics: from Setups and Algorithms to Toolboxes and Research Directions},
      author={Testa, Andrea and Carnevale, Guido and Notarstefano, Giuseppe},
      journal={arXiv preprint arXiv:2309.04257},
      year={2023}
    }

You can read our paper [here](https://arxiv.org/pdf/2309.04257).

*** -->

# Accompanying Video

<iframe width="260" height="115" src="https://www.youtube.com/embed/EckRNympXKs?si=a5gBMMLnQd_2cukk" title="Accompanying Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
