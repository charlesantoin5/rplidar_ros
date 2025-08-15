#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class RobotWallAlignment(Node):
    def __init__(self):
        super().__init__('robot_wall_alignment')
        
        # Paramètres robot
        self.robot_width = 0.40  # Robot fait 40cm de large
        self.detection_angle = 30  # Zone de détection ±45°
        self.min_wall_distance = 0.1  # Distance minimum du mur
        self.max_wall_distance = 2.0   # Distance maximum de détection
        self.obstacle_threshold = 0.03  # 3cm d'écart = obstacle
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Système d\'alignement robot-mur démarré')
        self.get_logger().info(f'Robot: {self.robot_width*100:.0f}cm de large')

    def scan_callback(self, msg):
        """Traitement principal pour l'alignement"""
        try:
            # Obtenir tous les points frontaux
            points = self.get_front_points(msg)
            
            if len(points) < 5:
                self.get_logger().warning('Pas assez de points pour détecter un mur')
                return
            
            # Détecter le mur principal
            wall_data = self.detect_main_wall(points)
            
            if wall_data:
                self.report_alignment_data(wall_data)
            else:
                self.get_logger().warning('Aucun mur détecté à l\'avant')
                
        except Exception as e:
            self.get_logger().error(f'Erreur: {str(e)}')

    def get_front_points(self, msg):
        """Extrait les points dans la zone frontale étendue"""
        points = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        total_points = len(msg.ranges)
        
        # Zone de détection étendue
        detection_rad = math.radians(self.detection_angle)
        index_range = int(detection_rad / angle_increment)
        
        # Points négatifs (fin du tableau)
        start_index = max(0, total_points - index_range)
        for i in range(start_index, total_points):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance, i))
        
        # Points positifs (début du tableau)
        for i in range(0, index_range + 1):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance, i))
        
        # Trier par angle
        points.sort(key=lambda p: p[2])
        return points

    def is_valid_distance(self, distance):
        """Vérifie la validité d'une distance"""
        return (self.min_wall_distance <= distance <= self.max_wall_distance and 
                not math.isinf(distance) and 
                not math.isnan(distance))

    def detect_main_wall(self, points):
        """Détecte le mur principal à l'avant du robot"""
        if len(points) < 3:
            return None
        
        # Trouver le groupe de points le plus proche de 0°
        best_segment = self.find_frontal_segment(points)
        
        if not best_segment or len(best_segment) < 3:
            return None
        
        # Calculer les caractéristiques du mur
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
        """Trouve le segment de mur le plus frontal"""
        # Grouper les points proches en distance (même mur)
        groups = []
        current_group = [points[0]]
        
        for i in range(1, len(points)):
            curr_dist = points[i][3]
            prev_dist = points[i-1][3]
            
            # Si la distance change beaucoup, nouveau groupe
            if abs(curr_dist - prev_dist) > 0.1:  # 10cm d'écart
                if len(current_group) >= 3:
                    groups.append(current_group)
                current_group = [points[i]]
            else:
                current_group.append(points[i])
        
        # Ajouter le dernier groupe
        if len(current_group) >= 3:
            groups.append(current_group)
        
        if not groups:
            return None
        
        # Choisir le groupe le plus proche de 0°
        best_group = None
        best_centering = float('inf')
        
        for group in groups:
            # Score de centrage (moyenne des angles absolus)
            centering = sum(abs(p[2]) for p in group) / len(group)
            if centering < best_centering:
                best_centering = centering
                best_group = group
        
        return best_group

    def calculate_wall_orientation(self, segment):
        """Calcule l'angle d'orientation du mur - STABLE avec régression linéaire"""
        if len(segment) < 3:
            return 0
        
        # Utiliser TOUS les points pour plus de stabilité (régression linéaire)
        x_coords = [p[0] for p in segment]
        y_coords = [p[1] for p in segment]
        n = len(segment)
        
        # Calculs de régression linéaire
        sum_x = sum(x_coords)
        sum_y = sum(y_coords)
        sum_xy = sum(x*y for x, y in zip(x_coords, y_coords))
        sum_x2 = sum(x*x for x in x_coords)
        
        # Éviter division par zéro
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-6:
            return 0
        
        # Pente de la ligne de régression
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Angle du mur par rapport à l'horizontal
        wall_angle = math.degrees(math.atan(slope))
        
        # Pour un robot qui avance vers un mur:
        # - Mur horizontal (perpendiculaire au robot) = 0°
        # - Mur qui monte vers la droite = angle positif
        # - Mur qui monte vers la gauche = angle négatif
        
        # Normaliser entre -90° et +90°
        while wall_angle > 90:
            wall_angle -= 180
        while wall_angle < -90:
            wall_angle += 180
    
        return wall_angle

    def calculate_wall_distance(self, segment):
        """Calcule la distance du point le plus proche à l'avant du robot"""
        # Trouver le point le plus proche de l'axe X (avant du robot)
        min_distance = float('inf')
        
        for point in segment:
            x, y, angle, distance = point[:4]
            # Distance du point à l'origine (position robot)
            point_distance = math.sqrt(x*x + y*y)
            if point_distance < min_distance:
                min_distance = point_distance
        
        return min_distance

    def calculate_linearity(self, segment):
        """Calcule l'écart moyen pour détecter obstacles"""
        if len(segment) < 3:
            return 0
        
        x_coords = [p[0] for p in segment]
        y_coords = [p[1] for p in segment]
        
        # Ligne de régression
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
        
        # Calculer l'écart de chaque point à la ligne
        total_deviation = 0
        for x, y in zip(x_coords, y_coords):
            expected_y = slope * x + intercept
            deviation = abs(y - expected_y)
            total_deviation += deviation
        
        return total_deviation / n

    def calculate_robot_coverage(self, segment):
        """Calcule quelle portion de la largeur robot est couverte"""
        if len(segment) < 2:
            return 0
        
        # Largeur du segment détecté
        first = segment[0]
        last = segment[-1]
        segment_width = math.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)
        
        # Pourcentage de couverture
        coverage = min(1.0, segment_width / self.robot_width)
        return coverage * 100  # En pourcentage

    def report_alignment_data(self, wall_data):
        """Affiche les données d'alignement pour le robot"""
        angle = wall_data['angle']
        if angle < 0:
            angle +=90
        elif angle >= 0:
            angle -=90
            
        distance = wall_data['distance']
        linearity = wall_data['linearity']
        coverage = wall_data['coverage']
        
        print('='*50)
        print('DONNÉES D\'ALIGNEMENT ROBOT-MUR')
        print(f'Angle mur:        {angle:+6.2f}° (0° = perpendiculaire)')
        print(f'Distance proche:  {distance:.3f}m')
        print(f'Couverture robot: {coverage:.1f}% ({self.robot_width*100:.0f}cm)')
        print(f'Linéarité:        {linearity*100:.1f}cm')
        
        # Instructions d'alignement
        if abs(angle) < 2:
            print('✅ ROBOT BIEN ALIGNÉ avec le mur')
        elif abs(angle) < 5:
            print(f'🔄 Ajustement mineur: {angle:+.1f}°')
        else:
            print(f'🔄 Correction nécessaire: {angle:+.1f}°')
        
        # Détection d'obstacles
        if linearity > self.obstacle_threshold:
            print('⚠️  OBSTACLES détectés sur le trajet')
        else:
            print('✅ Trajet LIBRE vers le mur')
        
        # Couverture
        if coverage < 50:
            print(f'⚠️  Couverture partielle ({coverage:.1f}% du robot)')
        else:
            print(f'✅ Bonne couverture de détection')
        
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