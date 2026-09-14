module App

open Elmish
open Elmish.React
open Fable.MaterialUI.Icons
open Fable.React
open Feliz
open Feliz.MaterialUI
open Feliz.prop
open Feliz.Router
open Fulma
open Utils

open AppState
open Checkface
open Logos
open Share
open Config


let parsePage = function
    | "retirement"::_ -> Retirement
    | _ -> Home

let parseUrl (path, query) =
    {
        Page = parsePage path
        FromValue = Map.tryFind "from_value" query
        ToValue = Map.tryFind "to_value" query
        FromGuid = Map.tryFind "from_guid" query |> Option.bind (System.Guid.TryParse >> tryToOption)
        ToGuid = Map.tryFind "to_guid" query |> Option.bind (System.Guid.TryParse >> tryToOption)
        FromSeed = Map.tryFind "from_seed" query |> Option.bind (System.UInt32.TryParse >> tryToOption)
        ToSeed = Map.tryFind "to_seed" query |> Option.bind (System.UInt32.TryParse >> tryToOption)
    }

let formatPathForVidValues vidValues =
    Feliz.Router.Router.formatPath("/",
        [
            match vidValues with
            | Some (fromValue, toValue) ->
                match fromValue with
                | CheckfaceValue value ->
                    "from_value", value
                | Guid guid ->
                    "from_guid", guid.ToString()
                | Seed seed ->
                    "from_seed", seed.ToString()

                match toValue with
                | CheckfaceValue value ->
                    "to_value", value
                | Guid guid ->
                    "to_guid", guid.ToString()
                | Seed seed ->
                    "to_seed", seed.ToString()
            | None ->
                ()
        ])

let oEmbedUrl (url:string) =
    oEmbedApiEndpoint + (Feliz.Router.Router.formatPath("", [ "url", url ]))

let homePageTitle vidValues =
    match vidValues with
    | Some (fromValue, toValue) ->
        sprintf "%s to %s" (shortCheckfaceSrcDesc fromValue) (shortCheckfaceSrcDesc toValue)
    | None ->
        siteName + " face morpher"

let homePageDescription = function
    | Some (fromValue, toValue) ->
            sprintf "Morph generated faces from %s to %s" (shortCheckfaceSrcDesc fromValue) (shortCheckfaceSrcDesc toValue)
    | None ->
            "Generate faces on the fly and morph between them"

let canonicalUrl (state:State) =
    match state.Page with
    | Home -> canonicalBaseUrl + formatPathForVidValues state.VidValues
    | Retirement -> canonicalBaseUrl + "/retirement"

let pageTitle (state:State) =
    match state.Page with
    | Home -> homePageTitle state.VidValues
    | Retirement -> $"API transition | %s{siteName}"

let pageDescription (state:State) =
    match state.Page with
    | Home -> homePageDescription state.VidValues
    | Retirement -> "Plans for retiring the current facemorph.me API and moving it to a new backend provider."

let getCurrentPath _ =
    Browser.Dom.window.location.pathname, Browser.Dom.window.location.search

let parseSegments (pathName, queryString) =
    let urlSegments = Router.urlSegments pathName RouteMode.Path
    let urlParams =
        (Router.createUrlSearchParams queryString).entries()
        |> Seq.map (fun arr -> (arr.[0], arr.[1]))
        |> Map.ofSeq
    urlSegments, urlParams

let chooseCheckfaceSrc value guid seed =
    match value, guid, seed with
    | Some value, _, _ -> Some (CheckfaceValue value)
    | None, Some guid, _ -> Some (Guid guid)
    | None, None, Some seed -> Some (Seed seed)
    | None, None, None -> None

let urlFromCheckfaceSrc urlState =
    chooseCheckfaceSrc urlState.FromValue urlState.FromGuid urlState.FromSeed

let urlToCheckfaceSrc urlState =
    chooseCheckfaceSrc urlState.ToValue urlState.ToGuid urlState.ToSeed

let defaultFromValue = Option.defaultValue (CheckfaceValue defaultTextValue)
let defaultToValue = Option.defaultWith (fun () -> CheckfaceValue (System.DateTime.Today.ToString("yyyy-MM-dd")))

let initByUrlstate urlState =
    let fromValue = urlFromCheckfaceSrc urlState
    let toValue = urlToCheckfaceSrc urlState
    let vidValues = Option.map2 (fun a b -> a,b) fromValue toValue
    {
        Page = urlState.Page
        LeftValue = defaultFromValue fromValue
        RightValue = defaultToValue toValue
        UploadDialogSide = None
        BrowseFacesDialogSide = None
        VidValues = vidValues
        ShareOpen = false
        ShareLinkMsg = None
        IsMorphLoading = false
        UseSlider = false
    }
let initByUrl = parseSegments >> parseUrl >> initByUrlstate
let init() = getCurrentPath() |> initByUrl, Cmd.none

let updateShare msg state =
    match msg with
    | OpenShare ->
        { state with ShareOpen = true }, Cmd.none
    | CloseShare ->
        { state with ShareOpen = false }, Cmd.none
    | CopyLink ->
        gtagEvent "CopyLink" "Share"

        state,
        match clipboard with
        | Some clipboard ->
            Cmd.OfPromise.either
                <| clipboard.writeText
                <| canonicalUrl state
                <| fun _ -> ShareMsg (SetShareMsg "Copied link to clipboard!")
                <| fun _ -> ShareMsg (SetShareMsg "Failed to copy to clipboard 😢")
        | None -> Cmd.ofMsg (ShareMsg (SetShareMsg "Failed to copy to clipboard 😢"))

    | NavigatorShare ->
        gtagEvent "NavigatorShareOpen" "Share"

        state,
        match navigatorCanShare, state.VidValues with
        | true, None ->
            Cmd.OfPromise.result (promise {
                ignore <| navigatorShare (canonicalUrl state) (pageTitle state) (pageDescription state)
                return ShareMsg (SetShareMsg "Sharing page!")
            })
        | true, Some vidValues ->
            Cmd.OfPromise.result (promise {
                let fileShare = {
                    FileName = homePageTitle state.VidValues + ".mp4"
                    Title = homePageTitle state.VidValues
                    FileUrl = vidSrc videoDim vidValues
                    Text = (homePageDescription state.VidValues)
                    Url = (canonicalUrl state)
                    ContentType = "video/mp4"
                }
                let! couldShareFile = shareFileFromUrl fileShare
                if couldShareFile then
                    return ShareMsg (SetShareMsg "Sharing morph!")
                else
                    return ShareMsg CopyLink
            })
        | false, _ ->
            Cmd.ofMsg (ShareMsg CopyLink)

    | SetShareMsg msg ->
        { state with ShareLinkMsg = Some msg },
        Cmd.OfPromise.result (promise {
            do! Promise.sleep 1000
            return ShareMsg ResetShareMsg
        })

    | ResetShareMsg ->
        { state with ShareLinkMsg = None }, Cmd.none

let update msg state : State * Cmd<Msg> =
    match msg with
    | SetLeftValue value ->
        { state with LeftValue = value }, Cmd.none
    | SetRightValue value ->
        { state with RightValue = value }, Cmd.none
    | ClickUploadRealImage side ->
        gtagEvent "OpenDialog" "EncodeImageDialog"
        { state with UploadDialogSide = Some side }, Cmd.none
    | CloseUploadDialog ->
        { state with UploadDialogSide = None }, Cmd.none
    | CloseBrowseFacesDialog ->
        { state with BrowseFacesDialogSide = None }, Cmd.none
    | BrowseCheckfaceValues side ->
        { state with BrowseFacesDialogSide = Some side }, Cmd.none
    | MakeVid when state.VidValues = Some (state.LeftValue, state.RightValue) ->
        state, Cmd.none
    | MakeVid ->
        gtagEvent "MakeVid" "Video"
        state, Some (state.LeftValue, state.RightValue) |> formatPathForVidValues |> Cmd.navigatePath
    | UrlChanged (path, query) ->
        let urlState = parseUrl (path, query)
        let urlFromSrc = urlFromCheckfaceSrc urlState
        let urlToSrc = urlToCheckfaceSrc urlState
        let vidValues = Option.map2 (fun a b -> a,b) urlFromSrc urlToSrc
        //changed and has something to be loading
        let isLoading = vidValues <> state.VidValues && vidValues.IsSome
        {
            state with
                Page = urlState.Page
                LeftValue = defaultFromValue urlFromSrc
                RightValue = defaultToValue urlToSrc
                VidValues = vidValues
                ShareOpen = false //always reset share state
                ShareLinkMsg = None
                IsMorphLoading = isLoading //set loading when has something to load
        }, Cmd.none
    | ShareMsg msg ->
        updateShare msg state
    | MorphLoaded ->
        { state with IsMorphLoading = false }, Cmd.none
    | SetUseSlider v ->
        let eventName = if v then "ToggleSliderOn" else "ToggleSliderOff"
        gtagEvent eventName "MorphSlider"
        { state with UseSlider = v }, Cmd.none

let getShareState (state:State) =
    {
        IsOpen = state.ShareOpen
        LinkMsg = state.ShareLinkMsg
        CanonicalUrl = canonicalUrl state
        IsShown = state.VidValues.IsSome
    }

let header =
    Html.header [
        prop.style [ style.marginBottom (length.em 2) ]
        prop.children [
            Column.column [ ] [
                Html.a [
                    prop.href "/"
                    prop.children [
                        Html.h1 [
                            prop.className "title "
                            prop.style [
                                style.custom ("fontSize", "clamp(2.5rem, 12vw, 4rem)")
                                style.display.flex
                                style.flexWrap.wrap
                                style.justifyContent.center
                                style.custom ("gap", length.em 0.5)
                            ]
                            prop.children [
                                Mui.svgIcon [
                                    prop.className "vivus-start"
                                    svgIcon.component' animatedLogo
                                    prop.style [
                                        style.fontSize (length.em 1)
                                    ]
                                ]
                                Html.span siteName
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]

let footer =
    Footer.footer [ CustomClass "app-footer" ] [
        Mui.container [
            container.maxWidth.md
            container.children [
                Html.div [
                    Html.div [
                        Html.p [
                            str "Contact: "
                            Mui.link [
                                link.color.initial
                                prop.href (sprintf "mailto:%s" contactEmail)
                                prop.text contactEmail
                            ]
                        ]
                        Html.p [
                            str "Source code: "
                            Mui.link [
                                link.color.initial
                                prop.href (sprintf "https://github.com/%s" githubRepo)
                                prop.children [
                                    gitHubIcon [ prop.style [ style.fontSize (length.em 1) ] ]
                                    str (" " + githubRepo)
                                ]
                            ]
                        ]
                        Html.p "If you find any bugs, please open an issue on GitHub"
                    ]
                    Html.div [
                        Mui.svgIcon [
                            svgIcon.component' logo
                            prop.style [
                                style.fontSize (length.rem 5)
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]

let retirementNotice =
    Html.aside [
        prop.className "retirement-notice"
        prop.children [
            Mui.container [
                container.maxWidth.md
                container.children [
                    Html.div [
                        prop.className "retirement-notice__content"
                        prop.children [
                            Html.div [
                                prop.className "retirement-notice__copy"
                                prop.children [
                                    Html.p [
                                        Html.strong "Update."
                                        str " We have pushed the facemorph.me API retirement back to "
                                        Html.span [
                                            prop.className "retirement-notice__date"
                                            prop.text apiTransitionDateLabel
                                        ]
                                        str "."
                                    ]
                                    Html.p [
                                        str "facemorph.me is staying up. We moved the date after a lot of you asked us to, so we can give the feedback the consideration it deserves. Hugging Face is still the leading candidate while we keep testing options."
                                    ]
                                ]
                            ]
                            Html.div [
                                prop.className "retirement-notice__actions"
                                prop.children [
                                    Mui.link [
                                        link.color.initial
                                        prop.href "/retirement"
                                        prop.text "Learn more"
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]

let createTheme isDark = [
    if isDark then theme.palette.type'.dark else theme.palette.type'.light
    theme.palette.background.default' <| if isDark then "#17181c" else "white"
    theme.palette.background.paper <| if isDark then "#353535" else "#fff"
    unbox<IThemeProp> ("palette.background.level2", if isDark then "#2a2a2a" else "#f5f5f5")

    let headingFamily = "'Nunito', 'Segoe UI', sans-serif"
    theme.typography.h1.fontFamily headingFamily
    theme.typography.h2.fontFamily headingFamily
    theme.typography.h3.fontFamily headingFamily
    theme.typography.h4.fontFamily headingFamily
    theme.typography.subtitle1.fontFamily headingFamily

    //everything else
    theme.typography.fontFamily "'Roboto', 'Segoe UI', sans-serif"

    theme.typography.body2.fontSize (length.rem 1)
]

let darkTheme = Styles.createMuiTheme (createTheme true)
let lightTheme = Styles.createMuiTheme (createTheme false)

[<ReactComponent>]
let ThemedApp children =
    let theme = if Hooks.useMediaQuery("@media (prefers-color-scheme: dark)") then darkTheme else lightTheme
    Mui.themeProvider [
        themeProvider.theme theme
        themeProvider.children children
    ]

let renderHome (state:State) (dispatch: Msg -> unit) =
    ThemedApp [
        Mui.cssBaseline [ ]
        Html.div [
            prop.className "page-with-retirement-notice"
            prop.children [
                header
                MorphForm.renderContent state dispatch
                MorphForm.renderEncodeImageDialog state dispatch
                MorphForm.renderBrowseFacesDialog state dispatch
                viewShareContent (getShareState state) (ShareMsg >> dispatch)
                Explain.view ()
                footer
            ]
        ]
        if not isSelfHost then retirementNotice
    ]

let renderRetirement () =
    ThemedApp [
        Mui.cssBaseline [ ]
        header
        Retirement.view ()
        footer
    ]

let render (state:State) (dispatch: Msg -> unit) =
    match state.Page with
    | Home -> renderHome state dispatch
    | Retirement -> renderRetirement ()

let inline helmet props = createElement (Fable.Core.JsInterop.import "Helmet" "react-helmet") props

let meta (property:string) content =
    Html.meta [
        prop.custom ("property", property)
        prop.custom ("content", content)
    ]

let viewHead (state:State) =
    let canonicalUrl = canonicalUrl state
    let title = pageTitle state
    let description = pageDescription state

    helmet [
        prop.children [
            Html.title (str title)
            Html.meta [
                prop.custom ("property", "description")
                prop.custom ("name", "description")
                prop.custom ("content", description)
            ]
            meta "og:title" title
            meta "og:description" description
            meta "og:site_name" siteName

            Html.link [ prop.rel.canonical; prop.href canonicalUrl ]
            meta "og:url" canonicalUrl

            match state.Page with
            | Home ->
                Html.link [
                    rel.alternate
                    prop.custom ("type", "application/json+oembed")
                    prop.href (oEmbedUrl canonicalUrl)
                    prop.title $"oEmbed for %s{title}"
                ]

                match state.VidValues with
                | None -> // just use left value if no video
                    meta "og:image" (imgSrc ogImgDim false state.LeftValue)
                    meta "og:image:alt" (imgAlt state.LeftValue)
                    meta "og:image:width" ogImgDim
                    meta "og:image:height" ogImgDim
                    meta "og:image:type" "image/jpeg"
                    meta "og:type" "website"

                | Some vidValues ->
                    let videoSrc = vidSrc ogVideoDim vidValues
                    let linkprevSrc = linkpreviewSrc linkpreviewWidth vidValues
                    let linkprevAlt = linkpreviewAlt vidValues

                    meta "og:image" linkprevSrc
                    meta "og:image:alt" linkprevAlt
                    meta "og:image:width" linkpreviewWidth
                    meta "og:image:height" linkpreviewHeight
                    meta "og:image:type" "image/jpeg"

                    meta "og:type" "video.other"
                    meta "og:video" videoSrc
                    meta "og:video:secure_url" videoSrc
                    meta "og:video:type" "video/mp4"
                    meta "og:video:width" ogVideoDim
                    meta "og:video:height" ogVideoDim
                    meta "twitter:card" "player"
                    meta "twitter:player" (videoSrc + "&embed_html=true")
                    meta "twitter:player:width" ogVideoDim
                    meta "twitter:player:height" ogVideoDim

            | Retirement ->
                meta "og:image" (imgSrc ogImgDim false state.LeftValue)
                meta "og:image:alt" (imgAlt state.LeftValue)
                meta "og:image:width" ogImgDim
                meta "og:image:height" ogImgDim
                meta "og:image:type" "image/jpeg"
                meta "og:type" "website"
        ]
    ]

[<ReactComponent>]
let RemoveServerSideJss () =
    // https://material-ui.com/guides/server-rendering/
    React.useEffectOnce <| fun () ->
        let jssStyles = Browser.Dom.document.getElementById "jss-server-side"
        if isTruthy jssStyles then
            jssStyles.parentElement.removeChild jssStyles |> ignore
    React.fragment [ ]

let viewWithoutRouter state dispatch =
    React.fragment [
        viewHead state
        RemoveServerSideJss ()
        render state dispatch
    ]


let view state dispatch =
    React.router [
        router.pathMode
        router.onUrlChanged (getCurrentPath >> parseSegments >> UrlChanged >> dispatch)
        router.children (viewWithoutRouter state dispatch)
    ]
