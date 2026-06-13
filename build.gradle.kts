import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.intellij.platform")
}

group = providers.gradleProperty("pluginGroup").get()
version = providers.gradleProperty("pluginVersion").get()

// NOTE: Repositories are configured centrally in settings.gradle.kts
// (dependencyResolutionManagement), as required by the IntelliJ Platform Gradle Plugin 2.x.

dependencies {
    intellijPlatform {
        // Build against PyCharm. Since 2025.3 (build 253) the Community/Professional split was
        // retired and a single `pycharm(...)` distribution is published, so the older
        // `pycharmCommunity(...)`/`pycharmProfessional(...)` helpers no longer resolve for
        // 2025.3+. The Python support plugin "PythonCore" — which provides PyClass,
        // PyClassType, PyCustomMember, TypeEvalContext, etc. — is bundled with it.
        pycharm(providers.gradleProperty("platformVersion").get())
        bundledPlugin("PythonCore")

        // In-IDE test fixtures (BasePlatformTestCase, CodeInsightTestFixture, ...).
        testFramework(TestFrameworkType.Platform)
    }

    // BasePlatformTestCase is a JUnit 4 test case.
    testImplementation("junit:junit:4.13.2")
}

// The platform test fixtures run on the IDE's own JUnit 4 runner.
tasks.test {
    useJUnit()
}

intellijPlatform {
    pluginConfiguration {
        ideaVersion {
            sinceBuild = providers.gradleProperty("pluginSinceBuild")
            untilBuild = providers.gradleProperty("pluginUntilBuild")
        }
    }
}

kotlin {
    // PyCharm 2024.3+ runs on JBR 21; build for the same target.
    jvmToolchain(21)
}
