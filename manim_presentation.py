# type: ignore
from manim_slides.slide import Slide, ThreeDSlide
from manimlib import (
    ABC,
    ALL_MODIFIERS,
    ARROW_SYMBOLS,
    ASPECT_RATIO,
    BLACK,
    BLUE,
    BLUE_A,
    BLUE_B,
    BLUE_C,
    BLUE_D,
    BLUE_E,
    BOLD,
    BOTTOM,
    CACHE_SIZE,
    CLOSED_THRESHOLD,
    COLOR_KEY,
    COLORMAP_3B1B,
    CURSOR_KEY,
    DARK_BROWN,
    DEFAULT_ANIMATION_LAG_RATIO,
    DEFAULT_ANIMATION_RUN_TIME,
    DEFAULT_ARROW_TIP_LENGTH,
    DEFAULT_ARROW_TIP_WIDTH,
    DEFAULT_BUFF_RATIO,
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_DASH_LENGTH,
    DEFAULT_DOT_RADIUS,
    DEFAULT_FILL_COLOR,
    DEFAULT_GLOW_DOT_RADIUS,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_LAGGED_START_LAG_RATIO,
    DEFAULT_LINE_SPACING_SCALE,
    DEFAULT_MOBJECT_TO_EDGE_BUFF,
    DEFAULT_MOBJECT_TO_MOBJECT_BUFF,
    DEFAULT_PIXEL_HEIGHT,
    DEFAULT_PIXEL_WIDTH,
    DEFAULT_RESOLUTION,
    DEFAULT_SMALL_DOT_RADIUS,
    DEFAULT_STROKE_COLOR,
    DEFAULT_STROKE_WIDTH,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    DEG,
    DEGREES,
    DL,
    DOWN,
    DR,
    EPSILON,
    ET,
    EVENT_DISPATCHER,
    FRAME_HEIGHT,
    FRAME_SHAPE,
    FRAME_WIDTH,
    FRAME_X_RADIUS,
    FRAME_Y_RADIUS,
    GOLD,
    GOLD_A,
    GOLD_B,
    GOLD_C,
    GOLD_D,
    GOLD_E,
    GRAB_KEY,
    GRAB_KEYS,
    GREEN,
    GREEN_A,
    GREEN_B,
    GREEN_C,
    GREEN_D,
    GREEN_E,
    GREEN_SCREEN,
    GREY,
    GREY_A,
    GREY_B,
    GREY_BROWN,
    GREY_C,
    GREY_D,
    GREY_E,
    IN,
    INFORMATION_KEY,
    ITALIC,
    LARGE_BUFF,
    LEFT,
    LEFT_SIDE,
    LIGHT_BROWN,
    LIGHT_PINK,
    MANIM_COLORS,
    MAROON,
    MAROON_A,
    MAROON_B,
    MAROON_C,
    MAROON_D,
    MAROON_E,
    MED_LARGE_BUFF,
    MED_SMALL_BUFF,
    NORMAL,
    NULL_POINTS,
    OBLIQUE,
    ORANGE,
    ORIGIN,
    OUT,
    PATH_TO_POINTS,
    PI,
    PINK,
    PROGRAM_UNIFORM_MIRRORS,
    PURPLE,
    PURPLE_A,
    PURPLE_B,
    PURPLE_C,
    PURPLE_D,
    PURPLE_E,
    RADIANS,
    RED,
    RED_A,
    RED_B,
    RED_C,
    RED_D,
    RED_E,
    RESIZE_KEY,
    RIGHT,
    RIGHT_SIDE,
    SCALE_FACTOR_PER_FONT_POINT,
    SELECT_KEY,
    SMALL_BUFF,
    STRAIGHT_PATH_THRESHOLD,
    SVG_HASH_TO_MOB_MAP,
    TAU,
    TEAL,
    TEAL_A,
    TEAL_B,
    TEAL_C,
    TEAL_D,
    TEAL_E,
    TEX_TO_SYMBOL_COUNT,
    TEXT_MOB_SCALE_FACTOR,
    TOP,
    TYPE_CHECKING,
    UL,
    UNSELECT_KEY,
    UP,
    UR,
    WHITE,
    X_AXIS,
    X_GRAB_KEY,
    Y_AXIS,
    Y_GRAB_KEY,
    YELLOW,
    YELLOW_A,
    YELLOW_B,
    YELLOW_C,
    YELLOW_D,
    YELLOW_E,
    Z_AXIS,
    AddTextWordByWord,
    AnimatedBoundary,
    AnimatedStreamLines,
    Animation,
    AnimationGroup,
    AnimationOnSurroundingRectangle,
    AnimationType,
    AnnularSector,
    Annulus,
    ApplyComplexFunction,
    ApplyFunction,
    ApplyMatrix,
    ApplyMethod,
    ApplyPointwiseFunction,
    ApplyPointwiseFunctionToCenter,
    ApplyWave,
    Arc,
    ArcBetweenPoints,
    Arrow,
    ArrowTip,
    Axes,
    BackgroundRectangle,
    BarChart,
    Brace,
    BraceLabel,
    BraceText,
    Broadcast,
    Bubble,
    BulletedList,
    Button,
    Cache,
    Camera,
    CameraFrame,
    ChangeDecimalToValue,
    ChangingDecimal,
    Checkbox,
    Checkmark,
    CheckpointManager,
    Circle,
    CircleIndicate,
    Clock,
    ClockPassesTime,
    Code,
    Color,
    ColorSliders,
    ComplexHomotopy,
    ComplexPlane,
    ComplexValueTracker,
    Cone,
    ControlMobject,
    ControlPanel,
    CoordinateSystem,
    CountInFrom,
    Cross,
    Cube,
    CubicBezier,
    CurvedArrow,
    CurvedDoubleArrow,
    CurvesAsSubmobjects,
    CyclicReplace,
    Cylinder,
    Dartboard,
    DashedLine,
    DashedVMobject,
    DecimalMatrix,
    DecimalNumber,
    DieFace,
    Difference,
    Disk3D,
    Dodecahedron,
    Dot,
    DotCloud,
    DoubleSpeechBubble,
    DrawBorderThenFill,
    Elbow,
    Ellipse,
    EnableDisableButton,
    EndScene,
    EventListener,
    EventType,
    Exclusion,
    ExitStack,
    Exmark,
    ExponentialValueTracker,
    Fade,
    FadeIn,
    FadeInFromPoint,
    FadeOut,
    FadeOutToPoint,
    FadeToColor,
    FadeTransform,
    FadeTransformPieces,
    Flash,
    FlashAround,
    FlashUnder,
    FlashyFadeIn,
    FocusOn,
    FullScreenFadeRectangle,
    FullScreenRectangle,
    FunctionGraph,
    Generic,
    GlowDot,
    GlowDots,
    Group,
    GrowArrow,
    GrowFromCenter,
    GrowFromEdge,
    GrowFromPoint,
    Homotopy,
    Image,
    ImageMobject,
    ImplicitFunction,
    Indicate,
    Integer,
    IntegerMatrix,
    InteractiveScene,
    InteractiveSceneEmbed,
    Intersection,
    Iterable,
    LaggedStart,
    LaggedStartMap,
    Laptop,
    LatexError,
    Lightbulb,
    Line,
    Line3D,
    LinearNumberSlider,
    MaintainPositionRelativeTo,
    MarkupText,
    Matrix,
    Mobject,
    MobjectMatrix,
    MotionMobject,
    MoveAlongPath,
    MoveToTarget,
    NumberLine,
    NumberPlane,
    OldSpeechBubble,
    OldThoughtBubble,
    OrderedDict,
    ParametricCurve,
    ParametricSurface,
    Path,
    PGroup,
    PhaseFlow,
    Piano,
    Piano3D,
    PMobject,
    Point,
    Polygon,
    Polyline,
    Prism,
    Prismify,
    ProgressDisplay,
    PygletWindow,
    PygletWindowKeys,
    R3_to_complex,
    Rectangle,
    RegularPolygon,
    ReplacementTransform,
    Restore,
    Rotate,
    Rotating,
    Rotation,
    RoundedRectangle,
    SampleSpace,
    ScaleInPlace,
    Scene,
    SceneFileWriter,
    SceneState,
    ScreenRectangle,
    Sector,
    SequenceMatcher,
    SGroup,
    ShaderWrapper,
    ShowCreation,
    ShowCreationThenDestruction,
    ShowCreationThenDestructionAround,
    ShowCreationThenFadeAround,
    ShowCreationThenFadeOut,
    ShowIncreasingSubsets,
    ShowPartial,
    ShowPassingFlash,
    ShowPassingFlashAround,
    ShowSubmobjectsOneByOne,
    ShrinkToCenter,
    SmallDot,
    SmoothedVectorizedHomotopy,
    SpeechBubble,
    Speedometer,
    Sphere,
    Square,
    Square3D,
    StreamLines,
    StringMobject,
    StrokeArrow,
    SubmobjectType,
    SubVmobjectType,
    Succession,
    Surface,
    SurfaceMesh,
    SurroundingRectangle,
    SVGMobject,
    Swap,
    TangentLine,
    Tex,
    TexMatrix,
    Text,
    Textbox,
    TexText,
    TexTextFromPresetString,
    TexturedSurface,
    ThoughtBubble,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
    Timer,
    TimeVaryingVectorField,
    TipableVMobject,
    Title,
    Torus,
    TracedPath,
    TracingTail,
    Transform,
    TransformFromCopy,
    TransformMatchingParts,
    TransformMatchingShapes,
    TransformMatchingStrings,
    TransformMatchingTex,
    Triangle,
    TrueDot,
    TurnInsideOut,
    TypeVar,
    Uncreate,
    Underline,
    Union,
    UnitInterval,
    UpdateFromAlphaFunc,
    UpdateFromFunc,
    ValueTracker,
    VCube,
    Vector,
    VectorField,
    VectorizedEarth,
    VectorizedPoint,
    VFadeIn,
    VFadeInThenOut,
    VFadeOut,
    VGroup,
    VGroup3D,
    VHighlight,
    VideoIcon,
    VideoSeries,
    VMobject,
    VMobjectFromSVGPath,
    VPrism,
    VShaderWrapper,
    VShowPassingFlash,
    WiggleOutThenIn,
    Window,
    Write,
    abstractmethod,
    adjacent_n_tuples,
    adjacent_pairs,
    always,
    always_redraw,
    always_rotate,
    always_shift,
    angle_axis_from_quaternion,
    angle_between_vectors,
    angle_of_vector,
    animation,
    annotations,
    appdirs,
    approx_smooth_quadratic_bezier_handles,
    arr_clip,
    array_is_constant,
    arrays_match,
    assert_is_mobject_method,
    average_color,
    batch_by_property,
    bezier,
    binary_search,
    cache_on_disk,
    camera,
    cartesian_product,
    cdist,
    center_of_mass,
    char_to_cahced_mob,
    choose,
    clear_cache,
    clip,
    clockwise_path,
    color_gradient,
    color_to_hex,
    color_to_int_rgb,
    color_to_int_rgba,
    color_to_rgb,
    color_to_rgba,
    compass_directions,
    complex_func_to_R3_func,
    complex_to_R3,
    config,
    constants,
    contextmanager,
    copy,
    counterclockwise_path,
    cross,
    cross2d,
    curve_to_quadratic,
    cycle_animation,
    deepcopy,
    diag_to_matrix,
    double_smooth,
    earclip_triangulation,
    earcut,
    event_handler,
    exponential_decay,
    extract_mobject_family_members,
    f_always,
    fdiv,
    find_file,
    find_intersection,
    full_range_specifier,
    gen_choose,
    get_cache_dir,
    get_closest_point_on_line,
    get_color_map,
    get_colormap_code,
    get_colormap_from_colors,
    get_colormap_list,
    get_directories,
    get_downloads_dir,
    get_full_raster_image_path,
    get_full_sound_file_path,
    get_full_vector_image_path,
    get_ipython,
    get_manim_dir,
    get_norm,
    get_num_args,
    get_output_dir,
    get_parameters,
    get_quadratic_approximation_of_cubic,
    get_raster_image_dir,
    get_rgb_gradient_function,
    get_sample_coords,
    get_shader_code_from_file,
    get_shader_dir,
    get_shader_program,
    get_smooth_cubic_bezier_handle_points,
    get_smooth_quadratic_bezier_path_through,
    get_sound_dir,
    get_temp_dir,
    get_unit_normal,
    get_vector_image_dir,
    get_vectorized_rgb_gradient_function,
    get_winding_number,
    gl,
    guarantee_existence,
    hash_obj,
    hash_string,
    hashlib,
    hex2rgb,
    hex_to_int,
    hex_to_rgb,
    image_path_to_texture,
    index_labels,
    inspect,
    int_to_hex,
    integer_interpolate,
    interpolate,
    interpolate_color,
    interpolate_color_by_hsl,
    inverse_interpolate,
    invert_color,
    invert_image,
    io,
    is_closed,
    is_inside_triangle,
    it,
    latex_to_svg,
    linalg,
    line_intersection,
    line_intersects_path,
    linear,
    linear_sum_assignment,
    lingering,
    list_difference_update,
    list_update,
    listify,
    log,
    logger,
    lru_cache,
    make_even,
    manim_config,
    manimpango,
    markup_to_svg,
    match_interpolate,
    math,
    merge_dicts_recursively,
    mglw,
    mid,
    midpoint,
    mobject,
    moderngl,
    module_loader,
    move_along_vector_field,
    move_points_along_vector_field,
    move_submobjects_along_vector_field,
    norm_squared,
    normalize,
    normalize_along_axis,
    not_quite_there,
    np,
    num_tex_symbols,
    numbers,
    ode_solution_points,
    op,
    os,
    outer_interpolate,
    override_animate,
    overshoot,
    partial_bezier_points,
    partial_quadratic_bezier_points,
    path_along_arc,
    pathops,
    pickle,
    pkg_resources,
    platform,
    plot_isoline,
    poly_line_length,
    prepare_animation,
    print_family,
    project_along_vector,
    pygments,
    pyperclip,
    pyplot,
    quadratic_bezier_points_for_arc,
    quaternion_conjugate,
    quaternion_from_angle_axis,
    quaternion_mult,
    random,
    random_bright_color,
    random_color,
    re,
    recursive_mobject_remove,
    reduce,
    register_font,
    remove_list_redundancies,
    remove_tex_environments,
    resize_array,
    resize_preserving_order,
    resize_with_interpolation,
    rgb2hex,
    rgb_to_color,
    rgb_to_hex,
    rgba_to_color,
    rotate_vector,
    rotate_vector_2d,
    rotation_about_z,
    rotation_between_vectors,
    rotation_matrix,
    rotation_matrix_from_quaternion,
    rotation_matrix_transpose,
    rotation_matrix_transpose_from_quaternion,
    running_start,
    rush_from,
    rush_into,
    scene,
    screeninfo,
    se,
    set_array_by_interpolation,
    set_program_uniform,
    shader_wrapper,
    shuffled,
    sigmoid,
    slow_into,
    smooth,
    smooth_quadratic_path,
    solve_ivp,
    square_to_cube_faces,
    squish_rate_func,
    straight_path,
    sys,
    tempfile,
    there_and_back,
    there_and_back_with_pause,
    thick_diagonal,
    time,
    tri_area,
    turn_animation_into_updater,
    utils,
    validators,
    vectorize,
    wiggle,
    window,
    wraps,
    z_to_vector,
)

title_text_kws = {
    "font_size": 48,
    "fill_color": "#333333",  # Dark grey color
}
body_text_kws = {
    "font_size": 24,
    "fill_color": "#333333",  # Dark grey color
}
sub_text_kws = {
    "font_size": 16,
    "fill_color": "#888888",  # Dark grey color
}


def detour_image(p, thresh=0.8):
    img = Image.open(p)
    detoured = Image.fromarray(
        np.where(
            np.prod(np.array(img) / 255, axis=2)[:, :, None] < thresh,
            np.array(img),
            np.array((0, 0, 0, 0), dtype=np.uint8),
        )
    )
    detoured.save(p + ".detoured.png")
    return p + ".detoured.png"


class MainSlide(Slide):
    def construct(self):
        # Set background color to beige
        self.camera.background_rgba = [*list(Color("#FAF8EE").rgb), 1.0]
        ## eg slide
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        # blabla
        self.next_slide()

        ## Title
        self.play(
            Write(Text("Self-supervised learning\nof dense visual representations", **title_text_kws).shift(3 * UP))  # type: ignore
        )
        self.play(Write(Text("PhD Thesis Defense", **body_text_kws).shift(DOWN * 2)))
        self.play(Write(Text("Timothée Darcet", **body_text_kws).shift(DOWN * 0.5)))
        self.next_slide()
        ## About me
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("About me", **title_text_kws).shift(3 * UP)))
        self.play(
            Write(
                Text(
                    "I am a PhD student at Inria (Grenoble) and Meta\n"
                    + "Advised by Maxime Oquab, Piotr Bojanowski and Julien Mairal (+ formerly Armand Joulin)\n"
                    + "Previously: master's at ENS Paris-Saclay (MVA) and master's at Ecole polytechnique\n"
                    + "I worked on: self-supervised learning, vision transformers",
                    **body_text_kws,
                ).shift(DOWN * 0.5)
            )
        )
        self.next_slide()
        ## Papers
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Papers", **title_text_kws).shift(3 * UP)))
        # Main papers
        main_papers_text = VGroup(
            Text("DINOv2: Learning Robust Visual Features without Supervision", **body_text_kws),
            Text(
                "Oquab*, Darcet*, Moutakanni*, et al, TMLR 2024 (featured cert. & outstanding cert. finalist)",
                **sub_text_kws,
            ),
            Text("Vision transformers need registers", **body_text_kws),
            Text("Darcet et al, ICLR 2024 (oral & outstanding paper award)", **sub_text_kws),
            Text("Cluster and Predict Latent Patches for Improved Masked Image Modeling", **body_text_kws),
            Text("Darcet et al, TMLR 2025", **sub_text_kws),
        )
        for i, (m1, m2) in enumerate(zip(main_papers_text.submobjects, main_papers_text.submobjects[1:], strict=False)):
            m2.next_to(m1, DOWN, aligned_edge=LEFT, buff=0.1 + 0.1 * (i % 2))
        main_papers_text.center().shift(UP)

        # Secondary papers
        # Secondary papers
        secondary_papers_text = VGroup(
            Text("Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach", **body_text_kws),
            Text("Vo et al, TMLR 2024", **sub_text_kws),
            Text(
                "DINOv2 Meets Text: A Unified Framework for Image-and Pixel-Level Vision-Language Alignment",
                **body_text_kws,
            ),
            Text("Jose et al, CVPR 2025", **sub_text_kws),
        )

        for i, (m1, m2) in enumerate(
            zip(secondary_papers_text.submobjects, secondary_papers_text.submobjects[1:], strict=False)
        ):
            m2.next_to(m1, DOWN, aligned_edge=LEFT, buff=0.1 + 0.1 * (i % 2))
        secondary_papers_text.next_to(main_papers_text, DOWN, aligned_edge=LEFT, buff=1)

        all_text = VGroup(main_papers_text, secondary_papers_text)
        all_text.center()

        main_papers_box = (
            Rectangle(
                width=max(main_papers_text.get_width(), secondary_papers_text.get_width()) + 0.5,
                # width=main_papers_text.get_width() + 0.5,
                height=main_papers_text.get_height() + 0.5,
            )
            .move_to(main_papers_text, aligned_edge=LEFT)
            .shift(LEFT * 0.25)
            .flip()
        )
        secondary_papers_box = (
            Rectangle(
                width=main_papers_box.get_width(),
                # width=secondary_papers_text.get_width() + 0.5,
                height=secondary_papers_text.get_height() + 0.5,
            )
            .move_to(secondary_papers_text)
            .flip()
        )

        self.play(Write(main_papers_text))
        self.play(ShowCreation(main_papers_box))
        self.play(Write(secondary_papers_text))
        self.play(ShowCreation(secondary_papers_box), lag_ratio=0.5)
        self.next_slide()
        ## Focus presented papers
        reg_title = Text("Registers", **body_text_kws).move_to(main_papers_text[2], aligned_edge=LEFT)
        capi_title = Text("CAPI", **body_text_kws).move_to(main_papers_text[4], aligned_edge=LEFT)

        self.play(*(FadeOut(mob) for mob in main_papers_text[:2] + secondary_papers_text))
        # self.play(*(FadeOut(mob) for mob in main_papers_text[3::2]))
        # self.play(FadeOut(secondary_papers_box), FadeOut(main_papers_box))
        self.play(Transform(main_papers_text[2], reg_title), Transform(main_papers_text[4], capi_title))
        self.next_slide()
        ## Registers: title
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Vision transformers need registers", **title_text_kws).shift(UP)))
        self.play(
            Write(
                Text("Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski", **sub_text_kws).shift(2 * DOWN)
            )
        )
        ## Primer: Vision transformer (ViT)
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Primer: Vision transformer (ViT)", **title_text_kws).to_edge(UP)))
        self.play(
            Write(
                VGroup(
                    Text("Simple and SOTA vision arch", **body_text_kws),
                    Text("Tokenizer = patchifier + linear", **body_text_kws),
                    Text("Use [CLS] token for output", **body_text_kws),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=1)
                .to_edge(LEFT)
            )
        )
        self.play(FadeIn(ImageMobject(detour_image("resources/ViT_diagram.png")).scale(1.3).to_edge(RIGHT)))
        self.play(
            Write(
                Text(
                    'Dosovitskiy et al. "An Image is Worth 16x16 Words: '
                    + 'Transformers for Image Recognition at Scale" ICLR 2020',
                    **sub_text_kws,
                ).to_edge(DR)
            )
        )
        ## Primer: attention maps
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Primer: attention maps", **title_text_kws).to_edge(UP)))
        self_attn_blk = VGroup(
            Rectangle(
                width=4,
                height=0.6,
            )
            .to_edge(RIGHT)
            .flip()
        )
        self_attn_blk.add(Text("Self-attention", **body_text_kws).move_to(self_attn_blk))
        self_attn_blk[0].set_fill(BLUE_A).set_opacity(1)
        token = VGroup(Rectangle(width=0.25, height=0.25, color="#000000").set_fill("#ffffff"))
        token.add(token[0].copy().shift(0.25 * UP))
        token.add(token[0].copy().shift(0.25 * DOWN))
        token.add(token[0].copy().shift(0.5 * DOWN))
        num_patches = 4
        tok_spacing = self_attn_blk.get_width() * RIGHT / (num_patches + 1)
        token.move_to(self_attn_blk.get_corner(UL), aligned_edge=DOWN).shift(tok_spacing * 0.5).shift(0.5 * UP)
        token.set_fill(GREEN_A).set_opacity(1)
        output_tokens = VGroup(token)
        for i in range(1, num_patches + 1):
            output_tokens.add(token.copy().shift(i * tok_spacing))
        imsize = 256
        side = 3
        pil_img = Image.open("resources/newjtx.jpg")
        w, h = pil_img.size
        pil_img = pil_img.resize((imsize, imsize), box=((w - h) / 2, 0, w - (w - h) / 2, h))
        patches = Group()
        for i in range(side):
            for j in range(side):
                # Crop the image into patches
                pil_img.crop(
                    (
                        i * imsize // side,
                        j * imsize // side,
                        (i + 1) * imsize // side,
                        (j + 1) * imsize // side,
                    )
                ).save(f"tmp/patch_{i}_{j}.png")
                patches.add(ImageMobject(f"tmp/patch_{i}_{j}.png").set_width(0.1).set_opacity(0))
        self.add(patches)
        permutation = np.argsort(np.random.rand(side**2))
        input_tokens = output_tokens.copy().shift((self_attn_blk.get_height() + 2) * DOWN)
        input_meanings = Group(
            Text("[CLS]", **body_text_kws).next_to(input_tokens[0], DOWN).shift(0.125 * DOWN),
            *(
                patches[permutation[i]].set_width(0.5).set_opacity(1).next_to(input_tokens[1 + i], DOWN)
                for i in range(num_patches)
            ),
        )
        permutation = np.concatenate(([0], permutation + 1))
        output_meanings = input_meanings.copy().next_to(output_tokens, UP)
        attention_lines = VGroup(
            *(
                VGroup(
                    *(
                        Line(
                            input_tokens[i].get_top(),
                            output_tokens[j].get_bottom(),
                            color=RED_A,
                            stroke_width=4,
                        )
                        for i in range(num_patches + 1)
                    )
                )
                for j in range(num_patches + 1)
            )
        )
        attention_lines.set_z_index(-1000)
        attn_mat = np.exp(np.eye(side**2 + 1) * 1 + np.random.randn(side**2 + 1, side**2 + 1) * 1)
        attn_mat = attn_mat / attn_mat.sum(axis=1, keepdims=True)
        attn_mat **= 0.5
        blk_diagram = Group(input_tokens, self_attn_blk, output_tokens, output_meanings, input_meanings)
        # self.add(input_tokens, self_attn_blk, output_tokens, output_meanings, input_meanings, attention_lines)
        self.play(ShowCreation(blk_diagram))
        ## attention lines
        recipe = VGroup(
            Text("- Compute the attention scores from [CLS]to patches", **body_text_kws),
            Text("- Reshape into a heatmap", **body_text_kws),
            Text("- Get an interpretable map of where the [CLS]token is “looking”", **body_text_kws),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        slide_text = (
            VGroup(
                Text("At every layer, each token attends to each token", **body_text_kws),
                Text("Attention map recipe:", **body_text_kws),
                recipe,
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            .to_edge(LEFT)
        )
        slide_text[-1].shift(0.1 * RIGHT)
        self.next_slide()
        self.play(
            ShowCreation(attention_lines),
            Write(slide_text[0]),
        )
        ## Focus CLS-->patches
        self.play(Write(slide_text[1]))

        normalizer = attn_mat[0, 1:].max()
        self.play(
            FadeOut(output_tokens[1:]),
            FadeOut(output_meanings[1:]),
            FadeOut(input_tokens[0]),
            FadeOut(input_meanings[0]),
            FadeOut(attention_lines[1:]),
            FadeOut(attention_lines[0][0]),
            *(
                attention_lines[i][j].animate().set_stroke(opacity=attn_mat[i, j] / normalizer)
                for i in range(1)
                for j in range(1, num_patches + 1)
            ),
        )
        self.play(Write(recipe[0]))
        ## Cleanup
        # self.remove(*(mob for mob in self.mobjects))
        ## Primer: reshape
        self.next_slide()
        self.play(FadeOut(input_tokens[1:]))
        self.play(
            # FadeInStay(patches),
            *(
                patches[i * side + j]
                .animate()
                .set_width(1)
                .set_opacity(1)
                .move_to(
                    blk_diagram.get_bottom() + 1.5 * UP + 0.5 * RIGHT + (side / 2 - 0.5) * LEFT + i * RIGHT + j * DOWN
                )
                if patches[i * side + j].get_width() > 0.25
                else patches[i * side + j]
                .move_to(
                    blk_diagram.get_bottom() + 1.5 * UP + 0.5 * RIGHT + (side / 2 - 0.5) * LEFT + i * RIGHT + j * DOWN
                )
                .animate()
                .set_width(1)
                .set_opacity(1)
                for j in range(side)
                for i in range(side)
            ),
        )
        self.play(Write(recipe[1]))
        ## Move CLS
        cls_tok = VGroup(output_tokens[0], output_meanings[0])
        patches.set_z_index(-30)
        attention_lines.set_z_index(-20)
        lines_end = (
            patches[side // 2].get_left()
            + 0.25 * LEFT
            + cls_tok.get_width() / 2 * LEFT
            + cls_tok.get_height() / 2 * DOWN
        )
        self.play(
            cls_tok.animate().next_to(patches[side // 2], LEFT, buff=0.25),
            *(
                attention_lines[i][j]
                .animate()
                .put_start_and_end_on(
                    patches[i * side + j].get_center(),
                    lines_end,
                )
                for i in range(1)
                for j in range(1, num_patches + 1)
            ),
            FadeOut(self_attn_blk),
        )
        ## Draw all attn lines
        image_attn_lines = VGroup(
            *(
                Line(
                    patches[i * side + j].get_center(),
                    lines_end,
                    color=RED_A,
                    stroke_width=4,
                )
                .set_z_index(-10)
                .set_opacity(attn_mat[0, i * side + j + 1] / normalizer)
                for i in range(side)
                for j in range(side)
            )
        )
        self.play(
            FadeOut(attention_lines[0][1:]),
            FadeIn(image_attn_lines),
        )
        ## Show attn map
        attmap = VGroup(
            *(
                Rectangle(1, 1, color=RED_A)
                .move_to(patches[i * side + j])
                .set_z_index(-10)
                .set_stroke(width=0)
                .set_opacity(attn_mat[0, i * side + j + 1] / normalizer)
                for i in range(side)
                for j in range(side)
            )
        )
        self.play(
            FadeOut(image_attn_lines),
            FadeIn(attmap),
        )
        self.play(Write(recipe[2]))
        self.play(
            FadeOut(cls_tok),
            attmap.animate.move_to(ORIGIN, coor_mask=[0, 1, 0]),
            patches.animate.move_to(ORIGIN, coor_mask=[0, 1, 0]),
        )
        ## The artifacts
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("The artifacts", **title_text_kws).to_edge(UP)))
        self.play(FadeIn(ImageMobject(detour_image("resources/yeet_attmaps.png")).scale(0.7).to_edge(DR)))
        self.play(
            Write(
                VGroup(
                    Text("Attention maps of ViTs have artifacts", **body_text_kws),
                    Text("Except DINO", **body_text_kws),
                    Text("But DINOv2 has them", **body_text_kws),
                    Text("?????", **title_text_kws),  # "???" meme?
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=1)
                .to_edge(LEFT)
            )
        )
        ## The norm outliers
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("The norm outliers", **title_text_kws).to_edge(UP)))
        self.play(
            Write(
                VGroup(
                    Text("Let's study DINOv2", **body_text_kws),
                    Text("We will need a criterion to say “is this token an artifact or not?”", **body_text_kws),
                    Text("We find a simple criterion: high norm", **body_text_kws),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
                .shift(UP)
                .to_edge(LEFT)
            )
        )
        self.play(FadeIn(ImageMobject(detour_image("resources/high_norms.png")).scale(0.7).to_edge(DR)))
        ## Where do those outliers appear?
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("When do those outliers appear?", **title_text_kws).to_edge(UP)))
        plot_1 = ImageMobject(detour_image("resources/norm_along_layers.png")).scale(0.7).to_edge(DL).shift(UP)
        caption_1 = (
            Text("In the ViT?\nAt layer >= 15", **body_text_kws, alignment="CENTER")
            .next_to(plot_1, UP)
            .shift(0.1 * LEFT)
        )
        self.play(Write(caption_1), FadeIn(plot_1))

        self.next_slide()
        plot_2 = ImageMobject(detour_image("resources/norm_along_iters.png")).scale(0.7).to_edge(DOWN).shift(UP)
        caption_2 = (
            Text("In the training?\nAround 150k iter", **body_text_kws, alignment="CENTER")
            .next_to(plot_2, UP)
            .shift(0.1 * LEFT)
        )
        self.play(Write(caption_2), FadeIn(plot_2))

        self.next_slide()
        plot_3 = ImageMobject(detour_image("resources/norm_along_iters.png")).scale(0.7).to_edge(DR).shift(UP)
        caption_3 = (
            Text("In which models?\n>= 300M parameters (ViT-L)", **body_text_kws, alignment="CENTER")
            .next_to(plot_3, UP)
            .shift(0.1 * LEFT)
        )
        self.play(Write(caption_3), FadeIn(plot_3))
        ## Where do those outliers appear?
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Where do those outliers appear?", **title_text_kws).to_edge(UP)))
        examples = ImageMobject(detour_image("resources/small_attmap.png")).to_edge(RIGHT).shift(UP)
        cos_sim_plot = ImageMobject(detour_image("resources/cosine_similarities.png"))
        self.play(FadeIn(examples))
        caption = (
            VGroup(
                Text("On patches that hold redundant information:", **body_text_kws),
                Text("ie patches that are very similar to their neighbors", **body_text_kws),
                Text("In practice: often background", **body_text_kws),
                Text("→ The model can discard this info without hurting representations", **body_text_kws),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(LEFT)
            .shift(UP)
        )
        self.play(Write(caption))
        self.play(FadeIn(cos_sim_plot.scale(1).next_to(caption, DOWN)))
        self.next_slide()
        ## What information do the high-norm tokens hold?
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(title := Text("What information do the high-norm tokens hold?", **title_text_kws).to_edge(UP)))
        self.play(Write(Text("(Compared to other patches)", **body_text_kws).next_to(title, DOWN)))
        self.play(
            Write(
                caption_1 := VGroup(
                    Text("Less local information", **body_text_kws, t2c={"Less": RED}),
                    Text("→ Model did discard this info!", **body_text_kws, t2c={"did": RED}),
                )
                .arrange(DOWN, aligned_edge=LEFT)
                .next_to(title, DOWN, aligned_edge=LEFT, buff=2)
            )
        )
        self.play(
            FadeIn(
                table_1 := ImageMobject(detour_image("resources/local_info_probing.png"))
                .scale(0.4)
                .next_to(caption_1, DOWN, aligned_edge=LEFT, buff=1.5)
            )
        )
        self.next_slide()
        self.play(
            FadeIn(
                table_2 := ImageMobject(detour_image("resources/global_info_probing.png"))
                .scale(0.4)
                .to_edge(RIGHT)
                .move_to(caption_1, coor_mask=[0, 1, 0])
            )
        )
        self.play(
            Write(
                VGroup(
                    Text("More global information", **body_text_kws, t2c={"More": RED}),
                    Text(
                        "→ These tokens are global information aggregators",
                        **body_text_kws,
                        t2c={"global information aggregators": RED},
                    ),
                )
                .arrange(DOWN, aligned_edge=RIGHT)
                .next_to(table_2, DOWN, aligned_edge=RIGHT)
                .move_to(table_1, coor_mask=[0, 1, 0])
            )
        )
        self.next_slide()
        ## The hypothesis
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        # self.play(Write(Text("The hypothesis", **title_text_kws).to_edge(UP)))
        self.play(
            Write(
                hypothesis := VGroup(
                    Text("Large, sufficiently trained models", **title_text_kws),
                    Text("learn to recognize redundant tokens", **title_text_kws),
                    Text("and to use them as places to store, process", **title_text_kws),
                    Text("and retrieve global information", **title_text_kws),
                )
                .arrange(DOWN)
                .center()
            )
        )
        self.play(
            ShowCreation(
                Rectangle(
                    width=hypothesis.get_width() + 1,
                    height=hypothesis.get_height() + 1,
                )
                .set_stroke(BLACK, width=10)
                .move_to(hypothesis),
                run_time=2,
            )
        )
        ## The fix: Registers
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("The fix: Registers", **title_text_kws).to_edge(UP)))
        self.play(
            Write(
                VGroup(
                    Text('Register := "useless token"', **body_text_kws),
                    Text("Input: learnable value (like [CLS])", **body_text_kws),
                    Text("Output: unused", **body_text_kws),
                    Text("They interact with other tokens through self-attention", **body_text_kws),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
                .to_edge(LEFT)
                .shift(UP * 1.5)
            )
        )
        self.play(FadeIn(ImageMobject(detour_image("resources/registers_diagram.png")).scale(1).to_edge(DOWN)))
        ## Does it work? (yes)
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Does it work?", **title_text_kws).to_edge(UP)))
        self.play(
            Write(
                caption := VGroup(
                    Text("Yes, it does!", **body_text_kws),
                    Text("We can train a model with registers", **body_text_kws),
                    Text("It has no artifacts", **body_text_kws),
                    Text("It has better representations", **body_text_kws),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
                .to_edge(LEFT)
                .shift(UP * 1)
            )
        )
        self.play(
            FadeIn(
                ImageMobject(detour_image("resources/pyrrhus_reg_before_after.png"))
                .scale(0.7)
                .to_edge(RIGHT)
                .move_to(caption, coor_mask=[0, 1, 0])
            )
        )
        self.play(FadeIn(ImageMobject(detour_image("resources/n_reg_score_curves.png")).scale(0.7).to_edge(DOWN)))
        ## What about supervised? And CLIP?
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(title := Text("What about supervised? And CLIP?", **title_text_kws).to_edge(UP)))
        caption = (
            VGroup(
                Text("It works too!", **body_text_kws),
                Text("Removes norm outliers", **body_text_kws),
                Text("Cleans up attention maps", **body_text_kws),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            .to_edge(LEFT)
            .shift(UP * 1)
        )
        self.play(Write(caption[0:2]))
        self.play(
            FadeIn(
                ImageMobject(detour_image("resources/norm_stripplots_before_after.png"))
                .scale(0.5)
                .to_edge(RIGHT)
                .next_to(caption, coor_mask=[0, 1, 0])
            )
        )
        self.next_slide()
        self.play(Write(caption[2:]))
        self.play(FadeIn(ImageMobject(detour_image("resources/pullfig_registers.png")).scale(0.9).to_edge(DOWN)))
        ## Bonus: attention maps of registers
        self.next_slide()
        self.play(*(FadeOut(mob) for mob in self.mobjects))
        self.play(Write(Text("Bonus: attention maps of registers", **title_text_kws).to_edge(UP)))
        reg_attmaps = ImageMobject(detour_image("resources/registers_attmaps.png")).scale(0.6).to_edge(DOWN)
        caption = VGroup(
            Text(
                "We can also visualize the attention from the registers to the patches",
                **body_text_kws,
                text2slant={"registers": "ITALIC"},
            ),
            Text("In some cases, different registers attend to different parts of the image", **body_text_kws),
            Text('Reminiscent of "slot attention" (Locatello et al. 2020)', **body_text_kws),
        ).arrange(DOWN, aligned_edge=LEFT)
        Group(caption, reg_attmaps).arrange(DOWN, aligned_edge=LEFT, buff=0.5).center()
        self.play(Write(caption))
        self.play(FadeIn(reg_attmaps))
