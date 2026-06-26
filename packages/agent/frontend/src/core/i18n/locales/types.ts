import type { LucideIcon } from "lucide-react";

export interface Translations {
  // Locale meta
  locale: {
    localName: string;
  };

  // Common
  common: {
    home: string;
    settings: string;
    delete: string;
    edit: string;
    rename: string;
    share: string;
    openInNewWindow: string;
    close: string;
    more: string;
    search: string;
    download: string;
    thinking: string;
    artifacts: string;
    public: string;
    custom: string;
    notAvailableInDemoMode: string;
    loading: string;
    loadMore: string;
    version: string;
    lastUpdated: string;
    code: string;
    preview: string;
    cancel: string;
    save: string;
    install: string;
    create: string;
    import: string;
    export: string;
    exportAsMarkdown: string;
    exportAsJSON: string;
    exportSuccess: string;
  };

  // Welcome
  welcome: {
    greeting: string;
    description: string;
    createYourOwnSkill: string;
    createYourOwnSkillDescription: string;
  };

  // Clipboard
  clipboard: {
    copyToClipboard: string;
    copiedToClipboard: string;
    failedToCopyToClipboard: string;
    linkCopied: string;
  };

  // Input Box
  inputBox: {
    placeholder: string;
    createSkillPrompt: string;
    addAttachments: string;
    tooManyFiles: string;
    attachmentImage: string;
    removeAttachment: string;
    stackLabel: string;
    stackFanOutTitle: string;
    mode: string;
    autoMode: string;
    autoModeDescription: string;
    flywheelMode: string;
    flywheelModeDescription: string;
    reasoningEffort: string;
    reasoningEffortMinimal: string;
    reasoningEffortMinimalDescription: string;
    reasoningEffortLow: string;
    reasoningEffortLowDescription: string;
    reasoningEffortMedium: string;
    reasoningEffortMediumDescription: string;
    reasoningEffortHigh: string;
    reasoningEffortHighDescription: string;
    searchModels: string;
    surpriseMe: string;
    surpriseMePrompt: string;
    followupLoading: string;
    followupConfirmTitle: string;
    followupConfirmDescription: string;
    followupConfirmAppend: string;
    followupConfirmReplace: string;
    suggestions: {
      suggestion: string;
      prompt: string;
      icon: LucideIcon;
    }[];
    suggestionsCreate: (
      | {
          suggestion: string;
          prompt: string;
          icon: LucideIcon;
        }
      | {
          type: "separator";
        }
    )[];
  };

  // Sidebar
  sidebar: {
    recentChats: string;
    newChat: string;
    chats: string;
    demoChats: string;
    agents: string;
    feedback: string;
  };

  // Agents
  agents: {
    title: string;
    description: string;
    newAgent: string;
    emptyTitle: string;
    emptyDescription: string;
    chat: string;
    delete: string;
    deleteConfirm: string;
    deleteSuccess: string;
    newChat: string;
    createPageTitle: string;
    createPageSubtitle: string;
    nameStepTitle: string;
    nameStepHint: string;
    nameStepPlaceholder: string;
    nameStepContinue: string;
    nameStepInvalidError: string;
    nameStepAlreadyExistsError: string;
    nameStepNetworkError: string;
    nameStepCheckError: string;
    nameStepBootstrapMessage: string;
    save: string;
    saving: string;
    saveRequested: string;
    saveHint: string;
    saveCommandMessage: string;
    agentCreatedPendingRefresh: string;
    more: string;
    agentCreated: string;
    startChatting: string;
    backToGallery: string;
  };

  // Breadcrumb
  breadcrumb: {
    workspace: string;
    chats: string;
  };

  // Workspace
  workspace: {
    officialWebsite: string;
    githubTooltip: string;
    settingsAndMore: string;
    visitGithub: string;
    reportIssue: string;
    contactUs: string;
    about: string;
  };

  // Conversation
  conversation: {
    noMessages: string;
    startConversation: string;
  };

  // Chats
  chats: {
    searchChats: string;
  };

  // Artifact gallery (spec phase0-3)
  gallery: {
    title: string;
    summaryCharts: (n: number) => string;
    summaryPerSubject: (n: number) => string;
    summaryAll: (n: number) => string;
    openGallery: (n: number) => string;
    openGalleryShort: string;
    downloadAll: (n: number) => string;
    downloadAllShort: string;
    exportDataTable: string;
    failedGenerated: (n: number) => string;
    failedReason: string;
    failedRemedy: string;
    noArtifacts: string;
    representative: string;
    reportTitle: string;
    reportOpen: string;
    reportDownload: string;
    backToChat: string;
    galleryPlaceholder: string;
    aggregate: string;
    perSubject: string;
    compareMode: string;
    exitCompare: string;
    downloadSelected: string;
    expandPerSubject: (n: number) => string;
    collapsePerSubject: string;
    clearFilters: string;
    filterParadigm: string;
    filterChartType: string;
    filterMode: string;
    filterGroup: string;
    filterSubject: string;
    allParadigms: string;
    allTypes: string;
    allModes: string;
    allGroups: string;
    allSubjects: string;
    searchPlaceholder: string;
    nFiltered: (n: number) => string;
  };

  // Page titles (document title)
  pages: {
    appName: string;
    chats: string;
    newChat: string;
    untitled: string;
  };

  // Tool calls
  toolCalls: {
    moreSteps: (count: number) => string;
    lessSteps: string;
    executeCommand: string;
    presentFiles: string;
    needYourHelp: string;
    useTool: (toolName: string) => string;
    searchForRelatedInfo: string;
    searchForRelatedImages: string;
    searchFor: (query: string) => string;
    searchForRelatedImagesFor: (query: string) => string;
    searchOnWebFor: (query: string) => string;
    viewWebPage: string;
    listFolder: string;
    readFile: string;
    writeFile: string;
    clickToViewContent: string;
    writeTodos: string;
    skillInstallTooltip: string;
    stageBroadcast: {
      dispatchSubagent: (subagentType: string) => string;
      parseHeaders: string;
      resolveCatalog: string;
      askClarification: string;
      runScript: (scriptName: string) => string;
      genericBash: string;
    };
  };

  // Uploads
  uploads: {
    uploading: string;
    uploadingFiles: string;
  };

  // Subtasks
  subtasks: {
    subtask: string;
    executing: (count: number) => string;
    in_progress: string;
    completed: string;
    failed: string;
    taskDescription: string;
    taskResult: string;
    expertWorking: string;
  };

  // Clarification
  clarification: {
    chooseOption: string;
    orTypeCustom: string;
    /** 决策卡标题前缀图标 + 强信号「分析已暂停」(spec#5 §3.1)。 */
    cardPausedTitle: string;
    /** risk_confirmation 专用更强标题 (spec#5 §3.5)。 */
    cardRiskTitle: string;
    /** 决策依据块前缀「为什么问：」(spec#5 §3.1，服务 feedback_identify_zone_info_not_persisted)。 */
    contextPrefix: string;
    /** 已答态闭环徽章「已确认」(spec#5 §3.1)。 */
    answeredBadge: string;
    /** 等待决策时输入框 placeholder (spec#5 §3.4)。 */
    awaitingPlaceholder: string;
    /** 等待决策时输入框旁轻提示 (spec#5 §3.4，可选)。 */
    awaitingHint: string;
    /** 选项键盘可达 a11y 提示「按数字键选择」(spec#5 §3.2)。 */
    keyboardHint: string;
  };

  // Run Trace（运行轨迹侧抽屉 —— spec 2026-06-24-frontend-phase0-2）
  runTrace: {
    /** 入口按钮 tooltip / aria-label */
    triggerLabel: string;
    /** 抽屉标题 */
    drawerTitle: string;
    /** 抽屉关闭按钮 aria-label（复用 common.close 也行，但单独留文案位更稳） */
    close: string;
    /** 空态：当前 run 还没有任何 agent 行为 */
    empty: string;
    /** 进行中徽章：「N 步进行中」 */
    runningSteps: (count: number) => string;
    /** 完成态徽章：「N 步」 */
    stepCount: (count: number) => string;
    /** 出错态徽章 */
    hasError: string;
    /** gate 节点标题（数据质量关卡） */
    gateTitle: string;
    /** gate 明细展开/收起 */
    showGateDetail: string;
    hideGateDetail: string;
    /** dispatch 节点展开/收起子步骤 */
    showSubSteps: string;
    hideSubSteps: string;
    /** 节点状态文字标签（色+图标+文字三件套） */
    statusRunning: string;
    statusOk: string;
    statusWarning: string;
    statusFailed: string;
    statusWaiting: string;
    /** 节点 kind 文字标签（用于图标旁的类别说明） */
    kindParadigm: string;
    kindDispatch: string;
    kindTool: string;
    kindGate: string;
    kindClarification: string;
    kindArtifact: string;
  };

  // Analysis Rail（分析进度轨 —— spec 2026-06-24-frontend-phase0-4）
  workflowStages: {
    /** nav 语义 aria-label（整条进度轨） */
    navLabel: string;
    /** 7 阶段名（key 与 WORKFLOW_STAGE_NAME_KEYS 对齐）+ charts 能力阶段（方案 B 动态轨） */
    names: {
      upload: string;
      paradigm: string;
      align: string;
      compute: string;
      qc: string;
      interpret: string;
      report: string;
      charts: string;
    };
    /** 阶段状态文字（色+图标+文字三件套 / color-not-only） */
    statusPending: string;
    statusActive: string;
    statusWaiting: string;
    statusDone: string;
    statusWarning: string;
    statusFailed: string;
    /** 等待 HITL 时的微标：「等你确认」 */
    waitingHint: string;
    /** tooltip：该阶段在做什么（hover/聚焦时） */
    whatItDoes: {
      upload: string;
      paradigm: string;
      align: string;
      compute: string;
      qc: string;
      interpret: string;
      report: string;
      charts: string;
    };
    /** 窄屏紧凑态：「第 N 阶段，共 total」（currentOrdinal 从 1 起；total 动态=本 run 显的阶段数） */
    compactOf: (currentOrdinal: number, total: number) => string;
    /** 窄屏紧凑态展开按钮 aria-label */
    compactExpand: string;
    compactCollapse: string;
  };

  // Token Usage
  tokenUsage: {
    title: string;
    label: string;
    input: string;
    output: string;
    total: string;
    view: string;
    unavailable: string;
    unavailableShort: string;
    note: string;
    presets: {
      off: string;
      summary: string;
      perTurn: string;
      debug: string;
    };
    presetDescriptions: {
      off: string;
      summary: string;
      perTurn: string;
      debug: string;
    };
    finalAnswer: string;
    stepTotal: string;
    sharedAttribution: string;
    subagent: (description: string) => string;
    startTodo: (content: string) => string;
    completeTodo: (content: string) => string;
    updateTodo: (content: string) => string;
    removeTodo: (content: string) => string;
  };

  // Shortcuts
  shortcuts: {
    searchActions: string;
    noResults: string;
    actions: string;
    keyboardShortcuts: string;
    keyboardShortcutsDescription: string;
    openCommandPalette: string;
    toggleSidebar: string;
  };

  // Settings
  settings: {
    title: string;
    description: string;
    sections: {
      account: string;
      appearance: string;
      memory: string;
      tools: string;
      skills: string;
      notification: string;
      about: string;
    };
    memory: {
      title: string;
      description: string;
      empty: string;
      rawJson: string;
      exportButton: string;
      exportSuccess: string;
      importButton: string;
      importConfirmTitle: string;
      importConfirmDescription: string;
      importFileLabel: string;
      importInvalidFile: string;
      importSuccess: string;
      manualFactSource: string;
      addFact: string;
      addFactTitle: string;
      editFactTitle: string;
      addFactSuccess: string;
      editFactSuccess: string;
      clearAll: string;
      clearAllConfirmTitle: string;
      clearAllConfirmDescription: string;
      clearAllSuccess: string;
      factDeleteConfirmTitle: string;
      factDeleteConfirmDescription: string;
      factDeleteSuccess: string;
      factContentLabel: string;
      factCategoryLabel: string;
      factConfidenceLabel: string;
      factContentPlaceholder: string;
      factCategoryPlaceholder: string;
      factConfidenceHint: string;
      factSave: string;
      factValidationContent: string;
      factValidationConfidence: string;
      noFacts: string;
      summaryReadOnly: string;
      memoryFullyEmpty: string;
      factPreviewLabel: string;
      searchPlaceholder: string;
      filterAll: string;
      filterFacts: string;
      filterSummaries: string;
      noMatches: string;
      markdown: {
        overview: string;
        userContext: string;
        work: string;
        personal: string;
        topOfMind: string;
        historyBackground: string;
        recentMonths: string;
        earlierContext: string;
        longTermBackground: string;
        updatedAt: string;
        facts: string;
        empty: string;
        table: {
          category: string;
          confidence: string;
          confidenceLevel: {
            veryHigh: string;
            high: string;
            normal: string;
            unknown: string;
          };
          content: string;
          source: string;
          createdAt: string;
          view: string;
        };
      };
    };
    appearance: {
      themeTitle: string;
      themeDescription: string;
      system: string;
      light: string;
      dark: string;
      systemDescription: string;
      lightDescription: string;
      darkDescription: string;
      languageTitle: string;
      languageDescription: string;
    };
    account: {
      profileTitle: string;
      email: string;
      role: string;
      changePasswordTitle: string;
      changePasswordDescription: string;
      currentPassword: string;
      newPassword: string;
      confirmNewPassword: string;
      passwordMismatch: string;
      passwordTooShort: string;
      passwordChangedSuccess: string;
      networkError: string;
      updating: string;
      updatePassword: string;
      signOut: string;
    };
    tools: {
      title: string;
      description: string;
    };
    skills: {
      title: string;
      description: string;
      createSkill: string;
      emptyTitle: string;
      emptyDescription: string;
      emptyButton: string;
    };
    notification: {
      title: string;
      description: string;
      requestPermission: string;
      deniedHint: string;
      testButton: string;
      testTitle: string;
      testBody: string;
      notSupported: string;
      disableNotification: string;
    };
    acknowledge: {
      emptyTitle: string;
      emptyDescription: string;
    };
  };
}
