#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class WallSegmentDetector(Node):
    def __init__(self):
        super().__init__('wall_segment_detector')
        
        # Paramètres de détection
        self.wall_width = 0.40  # Largeur du mur à détecter (40cm)
        self.front_angle_range = 40  # Zone de recherche ±30°
        self.min_points = 5  # Minimum de points pour valider un segment
        self.max_distance = 3.0  # Distance max de détection (3m)
        self.linearity_threshold = 0.02  # Seuil d'écart pour la linéarité (2cm)
        
        # Abonnement au topic LaserScan
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Détecteur de segment de mur démarré')
        self.get_logger().info(f'Largeur recherchée: {self.wall_width*100:.0f}cm')

    def scan_callback(self, msg):
        """Callback principal de traitement"""
        try:
            # Convertir les données polaires en cartésiennes
            points = self.polar_to_cartesian(msg)
            
            if not points:
                self.get_logger().warn('Aucun point valide détecté')
                return
            
            # Détecter le segment de mur
            wall_segment = self.detect_wall_segment(points)
            
            if wall_segment:
                self.analyze_wall_segment(wall_segment)
            else:
                self.get_logger().info('Aucun segment de mur détecté')
                
        except Exception as e:
            self.get_logger().error(f'Erreur: {str(e)}')

    def polar_to_cartesian(self, msg):
        """Convertit les données LaserScan en points cartésiens"""
        points = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        # Zone de recherche frontale
        total_points = len(msg.ranges)
        angle_range_rad = math.radians(self.front_angle_range)
        index_range = int(angle_range_rad / angle_increment)
        
        # Points autour de 0° (avant)
        start_index = max(0, total_points - index_range)
        end_index = min(total_points, index_range + 1)
        
        # Traiter les points de fin (angles négatifs)
        for i in range(start_index, total_points):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance))
        
        # Traiter les points de début (angles positifs)
        for i in range(0, end_index):
            distance = msg.ranges[i]
            if self.is_valid_distance(distance):
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append((x, y, angle, distance))
        
        # Trier par angle pour avoir une séquence continue
        points.sort(key=lambda p: p[2])  # Trier par angle
        return points

    def is_valid_distance(self, distance):
        """Vérifie si une distance est valide"""
        return (0.1 <= distance <= self.max_distance and 
                not math.isinf(distance) and 
                not math.isnan(distance))

    def detect_wall_segment(self, points):
        """Détecte un segment de mur de 40cm"""
        if len(points) < self.min_points:
            return None
        
        best_segment = None
        best_score = float('inf')
        
        # Fenêtre glissante pour trouver le meilleur segment
        for i in range(len(points) - self.min_points + 1):
            for j in range(i + self.min_points, len(points) + 1):
                segment = points[i:j]
                
                # Vérifier si la largeur correspond
                if self.calculate_segment_width(segment) >= self.wall_width * 0.8:  # Tolérance 20%
                    
                    # Calculer la linéarité
                    linearity_score = self.calculate_linearity(segment)
                    
                    if linearity_score < best_score and linearity_score < self.linearity_threshold:
                        best_score = linearity_score
                        best_segment = segment
        
        return best_segment

    def calculate_segment_width(self, segment):
        """Calcule la largeur physique d'un segment"""
        if len(segment) < 2:
            return 0
        
        first_point = segment[0]
        last_point = segment[-1]
        
        # Distance euclidienne entre premier et dernier point
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]
        width = math.sqrt(dx*dx + dy*dy)
        
        return width

    def calculate_linearity(self, segment):
        """Calcule l'écart moyen à la ligne droite (mesure de linéarité)"""
        if len(segment) < 3:
            return 0
        
        # Points de début et fin
        x1, y1 = segment[0][:2]
        x2, y2 = segment[-1][:2]
        
        # Éviter la division par zéro
        line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_length < 0.01:
            return float('inf')
        
        # Calculer l'écart de chaque point à la ligne droite
        total_deviation = 0
        for point in segment[1:-1]:  # Exclure les points d'extrémité
            x0, y0 = point[:2]
            
            # Distance point-ligne
            deviation = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / line_length
            total_deviation += deviation
        
        return total_deviation / max(1, len(segment) - 2)

    def analyze_wall_segment(self, segment):
        """Analyse et affiche les caractéristiques du segment de mur"""
        if not segment:
            return
        
        # Calculs principaux
        width = self.calculate_segment_width(segment)
        linearity = self.calculate_linearity(segment)
        
        # Point central du segment
        mid_index = len(segment) // 2
        center_point = segment[mid_index]
        center_x, center_y, center_angle, center_distance = center_point
        
        # Angle du mur par rapport au LIDAR
        first_point = segment[0]
        last_point = segment[-1]
        
        wall_vector_x = last_point[0] - first_point[0]
        wall_vector_y = last_point[1] - first_point[1]
        wall_angle = math.degrees(math.atan2(wall_vector_y, wall_vector_x))
        
        # Normaliser l'angle du mur (-90° à +90°)
        if wall_angle > 90:
            wall_angle -= 180
        elif wall_angle < -90:
            wall_angle += 180
        
        # Affichage des résultats
        self.get_logger().info('='*60)
        self.get_logger().info('SEGMENT DE MUR DÉTECTÉ')
        self.get_logger().info(f'  Points utilisés: {len(segment)}')
        self.get_logger().info(f'  Largeur mesurée: {width*100:.1f}cm')
        self.get_logger().info(f'  Distance du centre: {center_distance:.3f}m')
        self.get_logger().info(f'  Angle du centre: {math.degrees(center_angle):+.1f}°')
        self.get_logger().info(f'  Orientation du mur: {wall_angle:+.1f}°')
        self.get_logger().info(f'  Linéarité: {linearity*100:.1f}cm (écart moyen)')
        
        # Évaluation de la qualité
        if linearity < 0.01:
            quality = "EXCELLENTE"
        elif linearity < 0.02:
            quality = "BONNE"
        elif linearity < 0.03:
            quality = "CORRECTE"
        else:
            quality = "FAIBLE"
        
        self.get_logger().info(f'  Qualité de détection: {quality}')
        
        # Informations sur la position
        if abs(wall_angle) < 5:
            self.get_logger().info('  → Mur parallèle au LIDAR')
        elif wall_angle > 0:
            self.get_logger().info(f'  → Mur incliné vers la DROITE de {abs(90-wall_angle):.1f}°')
        else:
            self.get_logger().info(f'  → Mur incliné vers la GAUCHE de {abs(90+wall_angle):.1f}°')


def main(args=None):
    """Fonction principale"""
    rclpy.init(args=args)
    
    try:
        detector = WallSegmentDetector()
        rclpy.spin(detector)
        
    except KeyboardInterrupt:
        print('\nArrêt demandé par l\'utilisateur')
    except Exception as e:
        print(f'Erreur: {e}')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()