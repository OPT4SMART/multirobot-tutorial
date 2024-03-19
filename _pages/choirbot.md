---
layout: single
# classes: wide
author_profile: false
permalink: /choirbot
title: "ChoiRbot"
# defaults:
#   - scope:
#       type: pages
#     values:
toc: true
toc_sticky: true
---

## Introduction to ChoiRbot


**ChoiRbot** is a is a ROS 2 package to simulate and run experiments on teams of cooperating robots.


### Three-Layer architecture of ChoiRbot

**ChoiRbot** employs a three-layer architecture made of the _Team Guidance_, _RoboPlanning_, and _RoboControl_ layers. The _Team Guidance_ layer oversees high-level decision-making and robot lifecycle management, utilizing communication with neighboring entities for task execution. To this end, it is compatible with [DISROPT](disropt). The _RoboPlanning_ and _RoboControl_ layers handle lower-level control actions and receive setpoints from the guidance layer. Additionally, a dynamics integration layer called _RoboIntegration_ is provided. This layer facilitates integration, such as with `RViz` for visualization purposes.

<center>
<img src="{{site.baseurl}}/images/choirbot.png">
</center>




 <script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>