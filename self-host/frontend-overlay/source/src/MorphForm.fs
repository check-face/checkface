module MorphForm

open Fable.MaterialUI.Icons
open Fable.React
open Feliz
open Feliz.MaterialUI
open Fulma

open AppState
open Checkface
open FancyButton
open SliderMorph
open Config

let renderImageByValue value =
    //use source and srcset to allow picking webp if supported and to pick best size based on pixel scaling
    let src = imgSrc imgDim false value
    let srcSet isWebp =
        imgSizesSet
        |> List.map (fun dim -> sprintf "%s %iw" (imgSrc dim isWebp value) dim)
        |> String.concat ","
    let sizes = (sprintf "%ipx" imgDim) //displayed size is around imgDim in css px regardless of device scaling
    Html.picture [
        Html.source [
            prop.type' "image/webp"
            prop.custom ("srcSet", srcSet true)
            prop.sizes sizes
        ]
        Html.source [
            prop.type' "image/jpeg"
            prop.custom ("srcSet", srcSet false)
            prop.sizes sizes
        ]
        Html.img [
            if isSelfHost then prop.custom ("loading", "lazy")
            prop.src src
            prop.sizes sizes

            prop.width imgDim
            prop.height imgDim
            prop.alt (imgAlt value)
        ]
    ]

type private InputConfig = {
    Value : string
    Placeholder : string
    EndAdornment : ReactElement
    StartAdornment : ReactElement
    OnChange : string -> unit
    ExtraTextfieldProps : IReactProperty list
}

let tooltippedIconbutton (title: string) props =
    Mui.tooltip [
        tooltip.title title
        tooltip.children <|
            Mui.iconButton props
    ]

let private getInputConfig btnId value onChange onUploadRealImage onBrowseCheckfaceValues onClickMenu =
    let startAdornment hasFocusWithinClass =
        tooltippedIconbutton "Change Mode" [
            if hasFocusWithinClass then prop.className "focuswithin-child"
            iconButton.edge.start
            iconButton.children (menuIcon [])
            prop.onClick (ignore >> onClickMenu)
            prop.ariaLabel "change mode"
            prop.id (btnId:string)
        ]
    match value with
    | CheckfaceValue value ->
        {
            Value = value
            Placeholder = "Just type anything"
            StartAdornment = startAdornment true
            EndAdornment =
                tooltippedIconbutton "Browse Names" [
                    prop.className "focuswithin-child"
                    iconButton.edge.end'
                    iconButton.children (imageSearchIcon [])
                    prop.onClick (fun _ -> onBrowseCheckfaceValues ())
                    prop.ariaLabel "browse names"
                ]
            OnChange = CheckfaceValue >> onChange
            ExtraTextfieldProps = [
                textField.inputProps [
                    prop.style [ style.textAlign.center ]
                    prop.autoCapitalize.off
                ]
            ]
        }
    | Seed seed ->
        {
            Value = if seed = 0u then "" else string seed
            Placeholder = "Input an integer seed"
            StartAdornment = startAdornment true
            EndAdornment =
                React.fragment [ ] // don't need end adornment for numbers
            ExtraTextfieldProps = [
                textField.type' "number"
                textField.inputProps [
                    prop.min (float System.UInt32.MinValue)
                    prop.max (float System.UInt32.MaxValue)
                ]
            ]
            OnChange = fun value ->
                let onlyDigits =
                    value
                    |> String.filter System.Char.IsDigit

                match System.UInt32.TryParse onlyDigits with
                | _ when value <> onlyDigits -> ()
                | _ when value = "" -> onChange (Seed 0u) //treat empty string as 0
                | true, num -> onChange (Seed num)
                | _ -> () //do nothing - don't allow typing invalid number
                // there is somehow still a bug where it allows typing +,-,.,e
                // not sure how to fix that. System.Char.IsDigit is behaving propertly,
                // Maybe it's a bug in react?
        }
    | Guid guid ->
        {
            Value = shortCheckfaceSrcDesc (Guid guid)
            Placeholder = "Upload an image"
            StartAdornment = startAdornment false
            EndAdornment =
                tooltippedIconbutton "Upload Image" [
                    // don't use "focuswithin-child" with textField disabled
                    iconButton.edge.end'
                    iconButton.children (cameraIcon [])
                    prop.onClick (fun _ -> onUploadRealImage ())
                    prop.ariaLabel "upload image"
                ]
            OnChange = CheckfaceValue >> onChange //shouldn't happen because textField is disabled
            ExtraTextfieldProps = [
                textField.disabled true
            ]
        }

type SetpointKind =
    | TextValue
    | NumericSeed
    | SomeGuid

let setpointKindMenu (anchorEl:IRefValue<Option<Browser.Types.Element>>) isOpen (setMenuOpen: bool -> unit) setpointKind changeSetpointKind =
    Mui.menu [
        menu.anchorEl anchorEl
        menu.keepMounted true
        menu.open' isOpen
        menu.onClose (fun _ -> setMenuOpen false)
        menu.children [
            Mui.menuItem [
                menuItem.selected ((setpointKind = TextValue))
                prop.onClick (fun _ -> changeSetpointKind TextValue)
                menuItem.children [
                    Mui.listItemIcon [ textFieldsIcon [ ] ]
                    Mui.listItemText "Text Value"
                ]
            ]
            Mui.menuItem [
                menuItem.selected ((setpointKind = NumericSeed))
                prop.onClick (fun _ -> changeSetpointKind NumericSeed)
                menuItem.children [
                    Mui.listItemIcon [ dialpadIcon [ ] ]
                    Mui.listItemText "Numeric Seed"
                ]
            ]
            Mui.menuItem [
                menuItem.selected ((setpointKind = SomeGuid))
                prop.onClick (fun _ -> changeSetpointKind SomeGuid)
                menuItem.children [
                    Mui.listItemIcon [ cameraIcon [ ] ]
                    Mui.listItemText "Upload Image"
                ]
            ]
        ]
    ]

type private SetpointProps = {
    AutoFocus: bool
    Label: string
    Id: string
    Value: CheckfaceSrc
    OnChange: CheckfaceSrc -> unit
    OnUploadRealImage: unit -> unit
    OnBrowseCheckfaceValues: unit -> unit
}

[<ReactComponent>]
let private SetpointInput props =
    let anchorEl = React.useRef None
    let isMenuOpen, setMenuOpen = React.useState false

    let onClickMenu () = setMenuOpen true

    let changeModeBtnId = $"{props.Id}-changemodebtn"
    let inputId = $"{props.Id}-input"

    let inputConfig = getInputConfig changeModeBtnId props.Value props.OnChange props.OnUploadRealImage props.OnBrowseCheckfaceValues onClickMenu
    let setpointKind =
        match props.Value with
        | CheckfaceValue _ -> TextValue
        | Seed _ -> NumericSeed
        | Guid _ -> SomeGuid

    let changeSetpointKind kind =
        setMenuOpen false //close the menu
        match kind, props.Value with // to the most-reasonable new value
        | NumericSeed, AsNumericSeed seed -> props.OnChange (Seed seed)
        | TextValue, AsTextValue value -> props.OnChange (CheckfaceValue value)
        | SomeGuid, _ -> props.OnUploadRealImage ()

    Column.column [ ] [
        Html.div [
            prop.children [
                renderImageByValue props.Value
            ]
        ]

        Mui.noSsr [
            // the menu seems to throw "document undefined" in ssr
            setpointKindMenu (anchorEl :?> IRefValue<Option<Browser.Types.Element>>) isMenuOpen setMenuOpen setpointKind changeSetpointKind
        ]

        Mui.textField [
            textField.value inputConfig.Value

            textField.InputProps [
                prop.custom ("startAdornment",
                    Mui.inputAdornment [
                        inputAdornment.position.start
                        prop.ref anchorEl
                        prop.children [
                            inputConfig.StartAdornment
                        ]
                    ])
                prop.custom ("endAdornment",
                    Mui.inputAdornment [
                        inputAdornment.position.end'
                        prop.children [
                            inputConfig.EndAdornment
                        ]
                    ])
            ]
            yield! inputConfig.ExtraTextfieldProps

            textField.onChange inputConfig.OnChange
            textField.placeholder inputConfig.Placeholder
            textField.autoFocus props.AutoFocus
            prop.style [
                style.width (length.percent 100)
                style.maxWidth (length.px imgDim)
            ]
            prop.className "focuswithin-parent"
            textField.variant.outlined
            textField.label props.Label
            textField.margin.normal
            textField.id inputId
        ]
    ]

let renderMorph values useSlider dispatch =
    match values with
    | None -> Html.none
    | Some (fromValue, toValue) ->
        Html.div [
            if useSlider then
                ViewSliderMorph {
                    Values = (fromValue, toValue)
                    OnLoaded = (fun _ -> dispatch MorphLoaded)
                    FrameSrc = morphframeSrc
                    Dim = videoDim
                    NumFrames = 25
                }
            else
                let posterImgSrc = imgSrc imgDim false (fromValue) //imgDim is already in cache because fromValue is displayed at imgDim
                Html.img [
                    // A layout hack because video doesn't respect height attribute for auto sizing while loading,
                    // but img does. dummyImg is used to prevent vertical flickr while vid is loading (or if vid fails to load entirely).
                    // Do not place a src on this image because if it loads a broken image then it stops respecting
                    // auto sizing based on height prop. Once aspect-rasio css or intrinsicsize video attr are available this may not be necessary.
                    prop.width videoDim
                    prop.height videoDim
                    prop.className "morph-dummyImg"
                ]
                Feliz.Html.video [
                    prop.src (vidSrc videoDim (fromValue, toValue))
                    prop.controls true
                    prop.autoPlay true
                    prop.loop true
                    prop.muted true
                    prop.style [
                        style.display.block
                        style.margin.auto
                    ]
                    prop.className "morph-vid"
                    prop.poster posterImgSrc
                    prop.width videoDim
                    prop.height videoDim
                    prop.alt (vidMorphAlt (fromValue, toValue))
                    prop.onLoadedData (fun _ -> dispatch MorphLoaded)
                ]
            Mui.formControlLabel [
                prop.style [
                    style.display.block
                    style.marginLeft (length.px 5)
                    style.custom ("justifySelf", "center")
                    style.width (length.percent 100)
                    style.maxWidth (length.px videoDim)
                    style.height (length.px 0) //don't make morph button any lower
                ]
                formControlLabel.control (
                    Mui.checkbox [
                        checkbox.checked' useSlider
                        checkbox.onChange (SetUseSlider >> dispatch)
                    ]
                )
                formControlLabel.label "Use Slider"
            ]
        ]

let morphButton isLoading hideButton =
    Column.column [ ] [
        FancyButton [
            if hideButton && isLoading then
                prop.style [ style.opacity 0.7 ]
            elif hideButton then
                prop.style [ style.opacity 0. ]

            prop.type'.submit
            button.color.primary
            button.variant.contained
            button.size.large
            button.disabled (isLoading || hideButton)
            button.children [
                if isLoading then Mui.circularProgress [
                    circularProgress.size 20
                    circularProgress.color.inherit'
                    ] else str "Morph"
            ]
        ]
    ]

let renderContent (state:State) (dispatch: Msg -> unit) =
    Html.form [
        prop.onSubmit (fun e -> e.preventDefault(); dispatch MakeVid)
        prop.children [
            Mui.container [
                Html.div [
                    prop.className "morph-content"
                    prop.children [
                        SetpointInput {
                            AutoFocus = true
                            Value = state.LeftValue
                            Id = "leftval"
                            Label = "Morph from"
                            OnChange = (SetLeftValue >> dispatch)
                            OnUploadRealImage = (fun () -> ClickUploadRealImage Left |> dispatch)
                            OnBrowseCheckfaceValues = (fun () -> BrowseCheckfaceValues Left |> dispatch)
                        }
                        SetpointInput {
                            AutoFocus = false
                            Value = state.RightValue
                            Id = "rightval"
                            Label = "Morph to"
                            OnChange = (SetRightValue >> dispatch)
                            OnUploadRealImage = (fun () -> ClickUploadRealImage Right |> dispatch)
                            OnBrowseCheckfaceValues = (fun () -> BrowseCheckfaceValues Right |> dispatch)
                        }
                        let hideButton = state.VidValues = Some (state.LeftValue, state.RightValue)
                        morphButton state.IsMorphLoading hideButton
                        renderMorph state.VidValues state.UseSlider dispatch
                    ]
                ]
            ]
        ]
    ]

let renderEncodeImageDialog  state dispatch =
    EncodeImageDialog.EncodeImageDialog {
        OnClose = fun () -> dispatch CloseUploadDialog
        IsOpen = state.UploadDialogSide.IsSome
        RenderImgGuid = Guid >> renderImageByValue
        EncodeImageApiLocation = encodeApiAddr
        OnImageEncoded = fun guid ->
            dispatch CloseUploadDialog
            match state.UploadDialogSide with
            | Some Left -> SetLeftValue (Guid guid) |> dispatch
            | Some Right -> SetRightValue (Guid guid) |> dispatch
            | None -> ()
    }

let renderBrowseFacesDialog state dispatch =
    BrowseFacesDialog.BrowseFacesDialog {
        OnClose = fun () -> dispatch CloseBrowseFacesDialog
        IsOpen = state.BrowseFacesDialogSide.IsSome
        RenderValue = renderImageByValue
        OnValueSelected = fun value ->
            dispatch CloseBrowseFacesDialog
            match state.BrowseFacesDialogSide with
            | Some Left -> SetLeftValue value |> dispatch
            | Some Right -> SetRightValue value |> dispatch
            | None -> ()
        Values = Some [ for i in 1u..100u do Seed i ]
    }