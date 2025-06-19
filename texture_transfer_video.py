import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from meshpose.utils.detector_inference import PersonDetector
from meshpose.utils.meshpose_inference import MeshPoseInference
from meshpose.utils import round_np, visualize_vertices
from meshpose.postprocessing.mesh_renderer import MeshRenderer


def load_predictions_from_json(json_file):
    """Load model predictions from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_texture_image(texture_path):
    """
    Load texture image with transparency support
    """
    if not os.path.exists(texture_path):
        print(f"Texture file not found: {texture_path}")
        return None, None
    
    # Load image with alpha channel
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    
    if texture is None:
        print(f"Could not load texture image: {texture_path}")
        return None, None
    
    # Convert BGR to RGB
    if texture.shape[2] == 4:  # BGRA
        texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2RGBA)
        alpha_channel = texture[:, :, 3] / 255.0  # Normalize alpha to 0-1
        texture_rgb = texture[:, :, :3]
    elif texture.shape[2] == 3:  # BGR
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        texture_rgb = texture
        alpha_channel = np.ones((texture.shape[0], texture.shape[1]), dtype=np.float32)
    else:
        print(f"Unsupported texture format: {texture.shape}")
        return None, None
    
    return texture_rgb, alpha_channel


def apply_texture_overlay_to_bbox(image, bbox, texture_rgb, alpha_channel, blend_mode='normal'):
    """
    Apply texture as an overlay to the bounding box region, preserving the original texture appearance
    """
    result_image = image.copy().astype(np.float32)
    
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = image.shape[:2]
    
    # Clamp bounding box to image bounds
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return image
    
    # Resize texture to fit bounding box
    texture_resized = cv2.resize(texture_rgb, (bbox_width, bbox_height), interpolation=cv2.INTER_LANCZOS4)
    alpha_resized = cv2.resize(alpha_channel, (bbox_width, bbox_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply texture with alpha blending
    for y in range(bbox_height):
        for x in range(bbox_width):
            img_y = y1 + y
            img_x = x1 + x
            
            if 0 <= img_y < h and 0 <= img_x < w:
                alpha = alpha_resized[y, x]
                
                # Only apply texture where alpha is significant
                if alpha > 0.1:
                    texture_color = texture_resized[y, x]
                    
                    # Alpha blending
                    for c in range(3):  # RGB channels
                        result_image[img_y, img_x, c] = \
                            result_image[img_y, img_x, c] * (1 - alpha) + \
                            texture_color[c] * alpha
    
    return result_image.astype(np.uint8)


def apply_texture_to_person_silhouette(image, vertices, bbox, texture_rgb, alpha_channel):
    """
    Apply texture to the person's silhouette area defined by vertices
    """
    result_image = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    # Convert vertices to 2D screen coordinates
    vertices_2d = vertices[:, :2] if vertices.shape[1] >= 2 else vertices
    if vertices.shape[1] == 3:
        vertices_2d = vertices[:, :2]
    
    # Create mask from person vertices (convex hull)
    if len(vertices_2d) > 2:
        # Convert to integer coordinates
        vertices_int = vertices_2d.astype(np.int32)
        
        # Create convex hull for better silhouette
        hull = cv2.convexHull(vertices_int)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # Get bounding box of the hull
        x, y, mask_w, mask_h = cv2.boundingRect(hull)
        
        # Resize texture to fit the person's bounding area
        if mask_w > 0 and mask_h > 0:
            texture_resized = cv2.resize(texture_rgb, (mask_w, mask_h), interpolation=cv2.INTER_LANCZOS4)
            alpha_resized = cv2.resize(alpha_channel, (mask_w, mask_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply texture only within the person's silhouette
            for dy in range(mask_h):
                for dx in range(mask_w):
                    img_y = y + dy
                    img_x = x + dx
                    
                    if (0 <= img_y < h and 0 <= img_x < w and 
                        mask[img_y, img_x] > 0):  # Only within person silhouette
                        
                        alpha = alpha_resized[dy, dx]
                        
                        if alpha > 0.1:
                            texture_color = texture_resized[dy, dx]
                            
                            # Alpha blending
                            for c in range(3):
                                result_image[img_y, img_x, c] = \
                                    result_image[img_y, img_x, c] * (1 - alpha) + \
                                    texture_color[c] * alpha
    
    return result_image.astype(np.uint8)


def apply_texture_as_overlay_pattern(image, bbox, texture_rgb, alpha_channel, pattern_scale=0.5):
    """
    Apply texture as a repeating pattern overlay on the person
    """
    result_image = image.copy().astype(np.float32)
    
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = image.shape[:2]
    
    # Clamp bounding box
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return image
    
    # Scale texture for pattern effect
    pattern_width = int(texture_rgb.shape[1] * pattern_scale)
    pattern_height = int(texture_rgb.shape[0] * pattern_scale)
    
    if pattern_width > 0 and pattern_height > 0:
        texture_pattern = cv2.resize(texture_rgb, (pattern_width, pattern_height), interpolation=cv2.INTER_LANCZOS4)
        alpha_pattern = cv2.resize(alpha_channel, (pattern_width, pattern_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply pattern repeatedly across the bounding box
        for y in range(bbox_height):
            for x in range(bbox_width):
                img_y = y1 + y
                img_x = x1 + x
                
                if 0 <= img_y < h and 0 <= img_x < w:
                    # Calculate pattern coordinates (repeating)
                    pat_x = x % pattern_width
                    pat_y = y % pattern_height
                    
                    alpha = alpha_pattern[pat_y, pat_x]
                    
                    if alpha > 0.1:
                        texture_color = texture_pattern[pat_y, pat_x]
                        
                        # Reduced alpha for subtle overlay effect
                        blend_alpha = alpha * 0.7  # Make it more subtle
                        
                        for c in range(3):
                            result_image[img_y, img_x, c] = \
                                result_image[img_y, img_x, c] * (1 - blend_alpha) + \
                                texture_color[c] * blend_alpha
    
    return result_image.astype(np.uint8)


def apply_texture_to_mesh_fallback(image, vertices, bbox, texture_rgb=None, alpha_channel=None):
    """
    Fallback method for applying texture to mesh vertices (original approach with improvements)
    """
    if texture_rgb is None or alpha_channel is None:
        # Fall back to original bbox color extraction
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, min(w-1, int(x1)))
        y1 = max(0, min(h-1, int(y1)))
        x2 = max(0, min(w-1, int(x2)))
        y2 = max(0, min(h-1, int(y2)))
        
        if x2 > x1 and y2 > y1:
            texture_region = image[y1:y2, x1:x2]
            texture_color = np.mean(texture_region, axis=(0, 1))
        else:
            texture_color = [128, 128, 128]
        
        return texture_color, 1.0  # Full opacity
    
    # Use provided texture - sample from center for simplicity
    center_u, center_v = 0.5, 0.5
    h, w = texture_rgb.shape[:2]
    x = int(center_u * (w - 1))
    y = int(center_v * (h - 1))
    texture_color = texture_rgb[y, x]
    alpha = alpha_channel[y, x]
    
    return texture_color, alpha


def render_textured_vertices_fallback(image, vertices, texture_color, alpha=1.0, point_size=3):
    """
    Fallback method for rendering vertices with texture color overlay (improved version)
    """
    result_image = image.copy().astype(np.float32)
    
    # Convert vertices to image coordinates if needed
    vertices_2d = vertices[:, :2] if vertices.shape[1] >= 2 else vertices
    
    # Project 3D to 2D if vertices are 3D (simple orthographic projection)
    if vertices.shape[1] == 3:
        vertices_2d = vertices[:, :2]
    
    # Draw vertices with texture color
    for vertex in vertices_2d:
        x, y = int(vertex[0]), int(vertex[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Draw filled circle for vertex with alpha blending
            overlay = result_image.copy()
            cv2.circle(overlay.astype(np.uint8), (x, y), point_size, texture_color, -1)
            result_image = cv2.addWeighted(result_image.astype(np.uint8), 1-alpha, 
                                         overlay.astype(np.uint8), alpha, 0).astype(np.float32)
    
    return result_image.astype(np.uint8)


def process_video_with_texture(input_video, output_video, json_file, texture_path=None, application_mode='overlay'):
    """
    Process full video with texture application
    """
    # Load predictions from JSON
    predictions = load_predictions_from_json(json_file)
    
    if not predictions:
        print("No predictions found in JSON file")
        return
        
    # Open input video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Load texture if provided
    texture_rgb, alpha_channel = None, None
    if texture_path:
        texture_rgb, alpha_channel = load_texture_image(texture_path)
        if texture_rgb is not None:
            print(f"Loaded texture: {texture_path} ({texture_rgb.shape})")
        else:
            print("Warning: Failed to load texture")
    
    # Create detector for tracking
    detector = PersonDetector(momentum=0.6)
    
    # Process frames
    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing Video with Texture", unit="frame") as pbar:
        while cap.isOpened() and frame_idx < len(predictions):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[..., :3]
            
            # Get predictions for current frame
            frame_predictions = predictions[frame_idx]
            
            # Re-detect persons to get updated bounding boxes
            bboxes = detector(frame_rgb)
            
            # Match predictions with detections (use minimum of both)
            min_persons = min(len(bboxes), len(frame_predictions))
            
            result_image = frame_rgb.copy()
            
            for i in range(min_persons):
                person_data = frame_predictions[i]
                bbox = bboxes[i]
                
                # Extract vertices from JSON data
                vertices = None
                if 'xyz_hp' in person_data:
                    vertices = np.array(person_data['xyz_hp'])
                    vertex_type = 'xyz_hp'
                elif 'xyz_lp' in person_data:
                    vertices = np.array(person_data['xyz_lp'])
                    vertex_type = 'xyz_lp'
                else:
                    continue
                
                if texture_rgb is not None and alpha_channel is not None:
                    # Apply texture based on selected mode
                    if application_mode == 'overlay':
                        result_image = apply_texture_overlay_to_bbox(
                            result_image, bbox, texture_rgb, alpha_channel
                        )
                    elif application_mode == 'silhouette':
                        result_image = apply_texture_to_person_silhouette(
                            result_image, vertices, bbox, texture_rgb, alpha_channel
                        )
                    elif application_mode == 'pattern':
                        result_image = apply_texture_as_overlay_pattern(
                            result_image, bbox, texture_rgb, alpha_channel
                        )
                    elif application_mode == 'fallback':
                        # Use improved fallback method
                        texture_color, alpha = apply_texture_to_mesh_fallback(
                            frame_rgb, vertices, bbox, texture_rgb, alpha_channel
                        )
                        result_image = render_textured_vertices_fallback(
                            result_image, vertices, texture_color, alpha
                        )
                    else:
                        # Default to overlay
                        result_image = apply_texture_overlay_to_bbox(
                            result_image, bbox, texture_rgb, alpha_channel
                        )
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Write frame to output
            out.write(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Video with texture processed and saved to: {output_video}")


def process_video_with_json_mesh_renderer(input_video, output_video, json_file):
    """
    Process full video using JSON predictions with the original MeshRenderer
    """
    # Load predictions from JSON
    predictions = load_predictions_from_json(json_file)
    
    if not predictions:
        print("No predictions found in JSON file")
        return
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create mesh renderer
    renderer = MeshRenderer((width, height))
    
    # Process frames
    frame_idx = 0
    with tqdm(total=min(total_frames, len(predictions)), desc="Processing Video with Mesh Renderer", unit="frame") as pbar:
        while cap.isOpened() and frame_idx < len(predictions):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[..., :3]
            
            # Get predictions for current frame
            frame_predictions = predictions[frame_idx]
            
            # Extract vertices for all persons
            vertices_list = []
            for person_data in frame_predictions:
                if 'xyz_hp' in person_data:
                    vertices_list.append(np.array(person_data['xyz_hp']))
                elif 'xyz_lp' in person_data:
                    vertices_list.append(np.array(person_data['xyz_lp']))
            
            # Render frame with mesh renderer
            if vertices_list:
                result_image = renderer(frame_rgb, vertices_list)
            else:
                result_image = frame_rgb
            
            # Write frame to output
            out.write(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Video with mesh rendering processed and saved to: {output_video}")


def process_video(input_video, output_video, do_rendering=True):
    """Original video processing function"""
    # Open input video.
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Open output video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Create detector-tracking model.
    detector = PersonDetector(momentum=0.6)
    # Create MeshPose model.
    meshpose = MeshPoseInference()
    # Create mesh renderer.
    renderer = MeshRenderer((width, height)) if do_rendering else None

    model_predictions = list()
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[..., :3]
            bboxes = detector(frame)

            outputs = list()
            vertices = list()
            for bbox_ in bboxes:
                x1, y1, x2, y2 = bbox_
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                outputs_ = meshpose(frame, bbox_xywh)
                outputs_list_ = {key: round_np(item).tolist() for key, item in outputs_.items()}
                outputs.append(outputs_list_)
                vertices.append(outputs_['xyz_hp'])

            for bbox_ in bboxes:
                x1, y1, x2, y2 = bbox_.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if renderer is not None:
                frame = renderer(frame, vertices)
            else:
                frame = visualize_vertices(frame, outputs, vertices_type='xyz_lp')

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            model_predictions.append(outputs)
            pbar.update(1)

    cap.release()
    out.release()

    return model_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MeshPose video processing with texture overlay')
    parser.add_argument('--input_video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_dir', type=str, default='output_videos', help='Output directory')
    parser.add_argument('--do_rendering', action='store_true', help='Enable mesh rendering')
    parser.add_argument('--json_file', type=str, help='Path to JSON predictions file')
    parser.add_argument('--texture_image', type=str, help='Path to texture image (PNG/JPG with transparent background)')
    parser.add_argument('--process_json_only', action='store_true', 
                       help='Process video using only JSON predictions')
    parser.add_argument('--use_original_renderer', action='store_true',
                       help='Use original MeshRenderer with JSON data')
    parser.add_argument('--application_mode', type=str, default='overlay',
                       choices=['overlay', 'silhouette', 'pattern', 'fallback'],
                       help='Texture application mode: overlay (bbox), silhouette (person shape), pattern (repeating), fallback (vertex-based)')
    
    args = parser.parse_args()

    input_video = args.input_video
    video_name = os.path.basename(input_video).split('.')[0]
    output_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.process_json_only and args.json_file:
        # Process full video using JSON predictions
        if args.use_original_renderer:
            output_video = os.path.join(output_dir, f'{video_name}_json_mesh_rendered.mp4')
            process_video_with_json_mesh_renderer(input_video, output_video, args.json_file)
        else:
            if args.texture_image:
                texture_suffix = f"_preserved_{args.application_mode}"
            else:
                texture_suffix = "_json_processed"
            output_video = os.path.join(output_dir, f'{video_name}{texture_suffix}.mp4')
            process_video_with_texture(
                input_video, output_video, args.json_file, args.texture_image, args.application_mode
            )
    else:
        # Original video processing
        output_video = os.path.join(output_dir, f'{video_name}.mp4')
        output_model_predictions = os.path.join(output_dir, f'{video_name}.json')
        
        model_predictions = process_video(input_video, output_video, args.do_rendering)
        with open(output_model_predictions, 'w') as f:
            json.dump(model_predictions, f)
        
        print(f"Video processed and saved to: {output_video}")
        print(f"Predictions saved to: {output_model_predictions}")
        
    print("\n=== Usage Examples ===")
    print("1. Process video with overlay mode (logo on bounding box):")
    print(f"   python {os.path.basename(__file__)} --input_video video.mp4 --json_file predictions.json --texture_image netflix_logo.png --process_json_only --application_mode overlay")
    print("\n2. Process video with silhouette mode (logo on person shape):")
    print(f"   python {os.path.basename(__file__)} --input_video video.mp4 --json_file predictions.json --texture_image netflix_logo.png --process_json_only --application_mode silhouette")
    print("\n3. Process video with pattern mode (repeating logo):")
    print(f"   python {os.path.basename(__file__)} --input_video video.mp4 --json_file predictions.json --texture_image netflix_logo.png --process_json_only --application_mode pattern")
    print("\n4. Process video with fallback mode (improved vertex-based):")
    print(f"   python {os.path.basename(__file__)} --input_video video.mp4 --json_file predictions.json --texture_image netflix_logo.png --process_json_only --application_mode fallback")