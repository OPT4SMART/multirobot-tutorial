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

### Simulate a Robot
In **ChoiRbot**, robots can be simulated in two different ways. The first one is to leverage an external tool such as, e.g., Gazebo. The second one is to extend the `Integrator` class of the _RobotIntegration_ layer. We will focus on this second point.
Suppose that we are interested into simulating a robot subject to single-integrator dynamics in the form

$$
    \begin{align}
       \dot{x}(t) = u(t),
    \end{align}
$$

where $$x\in\mathbb{R}^{3}$$ is the 3D position of the robot while $$u\in\mathbb{R}^{3}$$ is a suitable control input. In the `Integrator` class, the position of the robot is stored in a variable called `current_pos`. The update of the dynamics is instead defined into an `integrate()` method. Thus, we have to specify how to update `current_pos`. Finally, we have to specify how to receive a control input. To this end, we have to define a ROS 2 subscription. A possible example for this class follows.

    from choirbot.integrator import Integrator
    from geometry_msgs.msg import Vector3
    import numpy as np


    class SingleIntegrator(Integrator):

        def __init__(self, integration_freq: float, odom_freq: float=None):
            super().__init__(integration_freq, odom_freq)

            # create input subscription
            self.u = np.zeros(3)
            self.subscription = self.create_subscription(Vector3, 'velocity', self.input_callback, 1)
            
            self.get_logger().info('Integrator {} started'.format(self.agent_id))
        
        def input_callback(self, msg):
            # save new input
            self.u = np.array([msg.x, msg.y, msg.z])

        def integrate(self):
            self.current_pos += self.samp_time * self.u

Here, we update the position using the Euler update. We receive inputs from a `velocity` topic. This class will publish an `Odometry` message into an _odom_ topic. All the machinery is already enclosed into the [`Integrator` class](https://github.com/OPT4SMART/ChoiRbot/blob/master/choirbot/choirbot/integrator/integrator.py). 

Notice that the class has a field named `agent_id`. This represents a unique numerical identifier for the robot and allows to instantiate a network of cooperating robots. This field also ensure uniqueness on the topic names by creating suitable namespaces for each robot. As an example, if `agent_id=0`, the robot will communicate on topic `/agent_0/odom` and `/agent_0/velocity`. 


### Implement a Control Law

Suppose now that we want to implement a proportional control law for the single-integrator system, i.e.

$$
    \begin{align}
       u(t) = K*(x(t)-x_d),
    \end{align}
$$

for a given goal $$x_d$$ and a suitable $$K$$. To do this, we can extend the `Controller` class of **ChoiRbot** as follows.


    from choirbot.controller import Controller
    from geometry_msgs.msg import Vector3, Point
    import numpy as np

    class ProportionalController(Controller):

        def __init__(self, pos_handler: str=None, pos_topic: str=None):
            
            super().__init__(pos_handler, pos_topic)
            self.pub= self.create_publisher(Vector3, 'velocity')     
            self.timer = self.create_timer(0.05, self.control_callback)
            self.goal = None
            self.goal_sub = self.create_subscription(Point,'goal', self.goal_callback, 1)
            self.get_logger().info('Controller {} started'.format(self.agent_id))
            
        def control_callback(self):
            # skip if position or goal is not available yet
            if self.current_pose.position is None or self.goal is None:
                return
            pos = self.current_pose.position
            k = -0.3
            v=k*np.array([pos-self.goal[0], pos-self.goal[1], pos-self.goal[2]])
            self.send_input(v)
        
        def send_input(self, v):
            msg = Vector3(v[0], v[1], v[2])
            self.pub.publish(msg)

        def goal_callback(self, msg):
            self.goal = [msg.x, msg.y, msg.z]
            self.get_logger().info('Received goal {}'.format(self.goal))
            
The `Controller` class already handles how to receive odometry data from the integrator or an external simulation tool. Thus, we only need to create a publisher to send input data. To this end, we created a _velocity_ topic. Notice that this name matches the one in the integrator class. The goal point will be received instead on a suitable _goal_ topic.


