#!/usr/bin/env python3
# fichier: robot_wall_alignment.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class RobotWallAlignment(Node):
    def __init__(self):
        super().__init__('robot_wall_alignment')
        
        self.robot_width = 0.40
        self.detection_angle = 30
        self.min_wall_distance = 0.1
        self.max_wall_distance = 2.0
        self.obstacle_threshold = 0.03
        self.target_distance = 0.40  # ✅ distance désirée: 40cm

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Système d\'alignement robot-mur démarré')
        self.get_logger().info(f'Robot: {self.robot_width*100:.0f}cm de large')
        self.get_logger().info(f'Distance cible au mur: {self.target_distance*100:.0f}cm')

    def scan_callback(self, msg):
        try:
            points = self.get_front_points(msg)
            if len(points) < 5:
                self.get_logger().warning('Pas assez de points pour détecter un mur')
                return
            
            wall_data = self.detect_main_wall(points)
            if wall_data:
                self.report_alignment_data(wall_data)
            else:
                self.get_logger().warning('Aucun mur détecté à l\'avant')
                
        except Exception as e:
            self.get_logger().error(f'Erreur: {str(e)}')

    def get_front_points(self, msg):
        points = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        total_points = len(msg.ranges)
        
        detection_rad = math.radians(self.detection_angle)
        index_range = int(detection_rad / angle_increment)
        
        start_index = max(0, total_points - index_range)
        for i in range(start_index, total_points):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance, i))
        
        for i in range(0, index_range + 1):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance, i))
        
        points.sort(key=lambda p: p[2])
        return points

    def is_valid_distance(self, distance):
        return (self.min_wall_distance <= distance <= self.max_wall_distance and 
                not math.isinf(distance) and 
                not math.isnan(distance))

    def detect_main_wall(self, points):
        if len(points) < 3:
            return None
        
        best_segment = self.find_frontal_segment(points)
        if not best_segment or len(best_segment) < 3:
            return None
        
        wall_angle = self.calculate_wall_orientation(best_segment)
        wall_distance = self.calculate_wall_distance(best_segment)
        linearity = self.calculate_linearity(best_segment)
        coverage = self.calculate_robot_coverage(best_segment)
        
        return {
            'segment': best_segment,
            'angle': wall_angle,
            'distance': wall_distance,
            'linearity': linearity,
            'coverage': coverage
        }

    def find_frontal_segment(self, points):
        groups = []
        current_group = [points[0]]
        
        for i in range(1, len(points)):
            curr_dist = points[i][3]
            prev_dist = points[i-1][3]
            if abs(curr_dist - prev_dist) > 0.1:
                if len(current_group) >= 3:
                    groups.append(current_group)
                current_group = [points[i]]
            else:
                current_group.append(points[i])
        
        if len(current_group) >= 3:
            groups.append(current_group)
        
        if not groups:
            return None
        
        best_group = None
        best_centering = float('inf')
        for group in groups:
            centering = sum(abs(p[2]) for p in group) / len(group)
            if centering < best_centering:
                best_centering = centering
                best_group = group
        return best_group

    def calculate_wall_orientation(self, segment):
        if len(segment) < 3:
            return 0
        x_coords = [p[0] for p in segment]
        y_coords = [p[1] for p in segment]
        n = len(segment)
        sum_x = sum(x_coords)
        sum_y = sum(y_coords)
        sum_xy = sum(x*y for x, y in zip(x_coords, y_coords))
        sum_x2 = sum(x*x for x in x_coords)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-6:
            return 0
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        wall_angle = math.degrees(math.atan(slope))
        while wall_angle > 90:
            wall_angle -= 180
        while wall_angle < -90:
            wall_angle += 180
        return wall_angle

    def calculate_wall_distance(self, segment):
        min_distance = float('inf')
        for point in segment:
            x, y, angle, distance = point[:4]
            point_distance = math.sqrt(x*x + y*y)
            if point_distance < min_distance:
                min_distance = point_distance
        return min_distance

    def calculate_linearity(self, segment):
        if len(segment) < 3:
            return 0
        x_coords = [p[0] for p in segment]
        y_coords = [p[1] for p in segment]
        n = len(segment)
        sum_x = sum(x_coords)
        sum_y = sum(y_coords)
        sum_xy = sum(x*y for x, y in zip(x_coords, y_coords))
        sum_x2 = sum(x*x for x in x_coords)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        total_deviation = 0
        for x, y in zip(x_coords, y_coords):
            expected_y = slope * x + intercept
            deviation = abs(y - expected_y)
            total_deviation += deviation
        return total_deviation / n

    def calculate_robot_coverage(self, segment):
        if len(segment) < 2:
            return 0
        first = segment[0]
        last = segment[-1]
        segment_width = math.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)
        coverage = min(1.0, segment_width / self.robot_width)
        return coverage * 100

    def report_alignment_data(self, wall_data):
        angle = wall_data['angle']
        if angle < 0:
            angle += 90
        else:
            angle -= 90
            
        distance = wall_data['distance']
        linearity = wall_data['linearity']
        coverage = wall_data['coverage']
        
        print('='*50)
        print('DONNÉES D\'ALIGNEMENT ROBOT-MUR')
        print(f'Angle mur:        {angle:+6.2f}° (0° = perpendiculaire)')
        print(f'Distance mesurée: {distance:.3f}m')
        print(f'Distance cible:   {self.target_distance:.3f}m')
        
        # ✅ comparaison avec la distance cible
        diff = distance - self.target_distance
        if abs(diff) <= 0.02:  # tolérance ±2cm
            print('✅ Distance correcte par rapport au mur (~40cm)')
        elif diff < 0:
            print(f'⚠️ Trop proche du mur ({abs(diff)*100:.1f}cm en moins)')
        else:
            print(f'⚠️ Trop éloigné du mur (+{diff*100:.1f}cm)')
        
        print(f'Couverture robot: {coverage:.1f}% ({self.robot_width*100:.0f}cm)')
        print(f'Linéarité:        {linearity*100:.1f}cm')
        
        if abs(angle) < 2:
            print('✅ ROBOT BIEN ALIGNÉ avec le mur')
        elif abs(angle) < 5:
            print(f'🔄 Ajustement mineur: {angle:+.1f}°')
        else:
            print(f'🔄 Correction nécessaire: {angle:+.1f}°')
        
        if linearity > self.obstacle_threshold:
            print('⚠️ OBSTACLES détectés sur le trajet')
        else:
            print('✅ Trajet LIBRE vers le mur')
        
        if coverage < 50:
            print(f'⚠️ Couverture partielle ({coverage:.1f}% du robot)')
        else:
            print('✅ Bonne couverture de détection')
        
        print(f'Distance sécuritaire: >{self.min_wall_distance*100:.0f}cm')


def main(args=None):
    rclpy.init(args=args)
    try:
        aligner = RobotWallAlignment()
        rclpy.spin(aligner)
    except KeyboardInterrupt:
        print('\nArrêt du système d\'alignement')
    finally:
        if 'aligner' in locals():
            aligner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
