use std::env;
use std::path::Path;

use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis};
use ort::{inputs, CPUExecutionProvider, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, AsImageView, WindowOptions};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

#[rustfmt::skip]
const CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[show_image::main]
fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    // ort::init()
    ort::init_from("/workspace/object_detection/rust/rtdetr_rust/libonnxruntime.so")
        // .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    // let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("baseball.jpg")).unwrap();

    let args: Vec<String> = env::args().collect();
    let image_file_path = args
        .get(1)
        .map_or("./img00343.png", |v| v);

    let original_img = image::open(image_file_path).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());

    // let model = Session::builder()?.commit_from_url(YOLOV8M_URL)?;
    let model = Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file("rtdetrv2_r18vd_120e_coco.onnx")?;

    use std::time::Instant;
    let start = Instant::now();

    // let img = original_img.resize_exact(640, 640, FilterType::CatmullRom); // CatmullRom is cubic
    let img = original_img.resize_exact(640, 640, FilterType::Triangle);
    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let mut orig_target_sizes = Array::zeros((1, 2));
    orig_target_sizes[[0, 0]] = img_width as i64;
    orig_target_sizes[[0, 1]] = img_height as i64;

    // Run RT-Detr inference
    let outputs: SessionOutputs = model.run(inputs!["images" => input.view(),
    "orig_target_sizes" => orig_target_sizes.view()
    ]?)?;
    let scores = outputs["scores"].try_extract_tensor::<f32>()?.into_owned();
    let labels = outputs["labels"].try_extract_tensor::<i64>()?.into_owned();
    let boxes = outputs["boxes"].try_extract_tensor::<f32>()?.into_owned();

    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    let mut results = Vec::new();
    for i in 0..300 {
        if scores[[0, i]] > 0.5 as f32 {
            println!(
                "{} {} {}",
                i,
                scores[[0, i]],
                CLASS_LABELS[labels[[0, i]] as usize]
            );
            results.push((
                BoundingBox {
                    x1: boxes[[0, i, 0]],
                    y1: boxes[[0, i, 1]],
                    x2: boxes[[0, i, 2]],
                    y2: boxes[[0, i, 3]],
                },
                CLASS_LABELS[labels[[0, i]] as usize],
                scores[[0, i]],
            ));
        }
    }

    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    for (bbox, label, _confidence) in results {
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        let color = match label {
            "person" => SolidSource {
                r: 0x00,
                g: 0x10,
                b: 0x80,
                a: 0x80,
            },
            "chair" => SolidSource {
                r: 0x20,
                g: 0x80,
                b: 0x40,
                a: 0x80,
            },
            _ => SolidSource {
                r: 0x80,
                g: 0x10,
                b: 0x40,
                a: 0x80,
            },
        };
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: f32::max(4., img_width as f32 / 800.),
                ..StrokeStyle::default()
            },
            &DrawOptions::new(),
        );
    }

    let overlay: show_image::Image = dt.into();

    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "ort + RT-Detr",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            window.set_image(
                "original_image",
                &original_img.as_image_view().map_err(|e| e.to_string())?,
            );
            window.set_overlay(
                "ai",
                &overlay.as_image_view().map_err(|e| e.to_string())?,
                true,
            );
            Ok(window.proxy())
        })
        .unwrap();

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}
