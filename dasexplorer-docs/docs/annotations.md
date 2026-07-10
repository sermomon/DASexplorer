# Annotation

Today DASexplorer supports four annotation types, each suited to different detection or labeling tasks. Each type is saved to its own CSV file, distinguished by a suffix appended to the data file's stem:

| Type | Use case | Docs | File suffix |
|---|---|---|---|
| BBox | Axis-aligned bounding boxes (e.g. object detection) | [BBox](annotation/bbox.md) | `_bbox.csv` |
| OBBox | Oriented bounding boxes (e.g. angled events) | [OBBox](annotation/obbox.md) | `_obb.csv` |
| Keypoints | Point-based labeling (e.g. call onsets) | [Keypoints](annotation/keypoints.md) | `_kp.csv` |
| Line | Line-based labeling (e.g. tracked trajectories) | [Line](annotation/line.md) | `_lin.csv` |

!!! note
    For example, annotating `myfile.h5` produces `myfile_bbox.csv`, `myfile_obb.csv`, `myfile_kp.csv`, and/or `myfile_lin.csv`, depending on which annotation types were used. This keeps annotations of different types separated and avoids naming collisions.

## Creating annotations

To annotate events, go to the **Annotate** section of the left panel and click one of the annotation types.

![Annotation panel](/img/screenshot-annotations-panel.png)

A crosshair guide following the pointer indicates that you are in edit mode. Draw the annotation on the visualization, then define an **ID** and a **comment** for it.

!!! tip
    Once an annotation is created, right-click it to bring up two options: **Edit Event** to modify its ID and comment, and **Edit Shape** to adjust its geometry.

To delete an existing annotation, right-click it and select **Remove Event**.

## Managing annotations

The **Annotations** section of the left panel lists all annotations currently present in the session.

From this list you can:

- **Save CSV** — export the current annotations to a CSV file
- **Clear All** — remove all current annotations

At the bottom of this section, the path where annotation files are saved is displayed. By default, this is the same directory as the data file, but you can change it at any time with the **...** button. If previous annotation files are found in this path, they will be displayed automatically. By convention, for each type of annotation (e.g., BBox), one CSV file is saved for each annotated DAS file.

## Next steps

- [Batch conversion](batch-conversion.md) to YOLO, COCO, or Raven formats